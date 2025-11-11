import os
import re
import time
import hashlib
import gradio as gr
import soundfile as sf
import numpy as np
from utils.logging import setup_logging, get_logger
from utils.normalize_text import VietnameseTTSNormalizer

# === Logging ===
setup_logging(run_name="vieneu-gradio", to_file=True, log_dir="logs", level="INFO")
log = get_logger("app.gradio")

log.info("Khởi động VieNeu-TTS (Gradio) ...")

from vieneu_tts.vieneu_tts import VieNeuTTS

# --- cấu hình thư mục output ---
OUTPUT_DIR = "output_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- cấu hình thư mục cache ref_codes ---
REF_CACHE_DIR = "cache_ref"
os.makedirs(REF_CACHE_DIR, exist_ok=True)
_REF_MEM: dict[str, np.ndarray] = {}  # cache trong RAM

# --- normalizer ---
TEXT_NORMALIZER = VietnameseTTSNormalizer()

def _slug(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s[:maxlen] if len(s) > maxlen else s

def _make_outpath(text: str, voice_choice: str | None) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    h8 = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:8]
    vslug = _slug(voice_choice or "custom")
    fname = f"{ts}_{vslug}_{h8}.wav"
    return os.path.join(OUTPUT_DIR, fname)

def _hash_file(path: str) -> str:
    try:
        st = os.stat(path)
        key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.md5((path or "none").encode("utf-8")).hexdigest()

def _cache_file_for_key(cache_key: str) -> str:
    # lưu trên đĩa theo md5 của cache_key, tránh ký tự lạ
    return os.path.join(REF_CACHE_DIR, hashlib.md5(cache_key.encode("utf-8")).hexdigest() + ".npy")

def _cache_get(cache_key: str) -> tuple[np.ndarray | None, bool]:
    # 1) RAM
    arr = _REF_MEM.get(cache_key)
    if arr is not None:
        return arr, True
    # 2) Đĩa
    fpath = _cache_file_for_key(cache_key)
    if os.path.exists(fpath):
        try:
            arr = np.load(fpath)
            _REF_MEM[cache_key] = arr
            return arr, True
        except Exception as e:
            log.warning("Không đọc được cache đĩa %s: %s", fpath, e)
    return None, False

def _cache_put(cache_key: str, arr: np.ndarray) -> None:
    _REF_MEM[cache_key] = arr
    fpath = _cache_file_for_key(cache_key)
    try:
        # ép int32 cho an toàn
        np.save(fpath, arr.astype(np.int32, copy=False))
    except Exception as e:
        log.warning("Không ghi được cache đĩa %s: %s", fpath, e)

tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    codec_repo="neuphonic/neucodec",
)
log.info("Model đã tải xong")

# === Danh sách giọng mẫu ===
VOICE_SAMPLES = {
    "Nam 1 (id_0001)": {"audio": "./sample/id_0001.wav", "text": "./sample/id_0001.txt"},
    "Nữ 1 (id_0002)": {"audio": "./sample/id_0002.wav", "text": "./sample/id_0002.txt"},
    "Nam 2 (id_0003)": {"audio": "./sample/id_0003.wav", "text": "./sample/id_0003.txt"},
    "Nữ 2 (id_0004)": {"audio": "./sample/id_0004.wav", "text": "./sample/id_0004.txt"},
    "Nam 3 (id_0005)": {"audio": "./sample/id_0005.wav", "text": "./sample/id_0005.txt"},
    "Nam 4 (id_0007)": {"audio": "./sample/id_0007.wav", "text": "./sample/id_0007.txt"},
}

def _resolve_ref(voice_choice: str, custom_audio: str | None, custom_text: str | None):
    """
    Trả về (ref_audio_path, ref_text, cache_key, src_desc)
    cache_key: xác định duy nhất theo preset/custom + hash file
    """
    if custom_audio and custom_text:
        audio_path = custom_audio
        ref_text = custom_text
        cache_key = f"custom:{_hash_file(audio_path)}"
        src_desc = "giọng tùy chỉnh"
        return audio_path, ref_text, cache_key, src_desc

    if voice_choice in VOICE_SAMPLES:
        audio_path = VOICE_SAMPLES[voice_choice]["audio"]
        text_path = VOICE_SAMPLES[voice_choice]["text"]
        with open(text_path, "r", encoding="utf-8") as f:
            ref_text = f.read()
        cache_key = f"preset:{voice_choice}:{_hash_file(audio_path)}"
        src_desc = f"giọng mẫu {voice_choice}"
        return audio_path, ref_text, cache_key, src_desc

    raise ValueError("Vui lòng chọn giọng hoặc tải audio + text tùy chỉnh.")

def _load_ref(voice_choice: str, custom_audio: str | None, custom_text: str | None):
    """
    Trả về (ref_codes(np.ndarray[int32]), ref_text(str), cache_key(str), is_cache_hit(bool))
    """
    ref_audio, ref_text, cache_key, src_desc = _resolve_ref(voice_choice, custom_audio, custom_text)

    # 1) thử lấy từ cache
    arr, hit = _cache_get(cache_key)
    if hit and isinstance(arr, np.ndarray) and arr.ndim == 1:
        log.info("REF CACHE HIT: %s → %s tokens", src_desc, arr.shape[0])
        return arr, ref_text, cache_key, True

    # 2) không có cache → encode
    log.info("REF CACHE MISS: %s → encode_reference()", src_desc)
    t0 = time.perf_counter()
    ref_codes_tensor = tts.encode_reference(ref_audio)  # có thể là torch.Tensor
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # ép về np.ndarray[int32] để cache, tránh chiếm VRAM
    if hasattr(ref_codes_tensor, "detach"):
        ref_codes_np = ref_codes_tensor.detach().cpu().numpy().astype(np.int32, copy=False)
    elif isinstance(ref_codes_tensor, np.ndarray):
        ref_codes_np = ref_codes_tensor.astype(np.int32, copy=False)
    else:
        # list-like
        ref_codes_np = np.asarray(list(ref_codes_tensor), dtype=np.int32)

    log.info("encode_reference() xong: %.1f ms, codes=%s", dt_ms, tuple(ref_codes_np.shape))
    _cache_put(cache_key, ref_codes_np)
    return ref_codes_np, ref_text, cache_key, False

# --- preview normalize (hiển thị sau chuẩn hoá) ---
def preview_normalize(text: str, use_normalizer: bool) -> str:
    if use_normalizer and text:
        try:
            return TEXT_NORMALIZER.normalize(text)
        except Exception as e:
            log.exception("Lỗi normalize preview: %s", e)
            return ""
    return ""

def synthesize_speech(text, voice_choice, custom_audio=None, custom_text=None, use_normalizer: bool = False):
    try:
        if not text or text.strip() == "":
            return None, "❌ Vui lòng nhập văn bản cần tổng hợp", ""

        if len(text) > 250:
            return None, "❌ Văn bản quá dài! Vui lòng nhập tối đa 250 ký tự", ""

        # nếu bật normalizer -> dùng text đã chuẩn hoá
        text_to_use = TEXT_NORMALIZER.normalize(text) if use_normalizer else text
        norm_preview = text_to_use if use_normalizer else ""

        ref_codes, ref_text, cache_key, hit = _load_ref(voice_choice, custom_audio, custom_text)

        log.info("Infer: len(text)=%d, cache_hit=%s, normalized=%s",
                 len(text_to_use), hit, use_normalizer)
        t0 = time.perf_counter()
        wav = tts.infer(text_to_use, ref_codes, ref_text)
        dt = (time.perf_counter() - t0) * 1000
        log.info("infer() xong: %.1f ms", dt)

        out_path = _make_outpath(text_to_use, voice_choice)
        sf.write(out_path, wav, tts.sample_rate)
        log.info("Lưu file: %s", out_path)

        cache_str = "cache=hit" if hit else "cache=miss"
        norm_str = "normalize=on" if use_normalizer else "normalize=off"
        return out_path, f"✅ Tổng hợp thành công! {cache_str} | {norm_str} | time: {dt:.1f} ms", norm_preview
    except Exception as e:
        log.exception("Lỗi synthesize_speech: %s", e)
        return None, f"❌ Lỗi: {str(e)}", ""

examples = [
    ["Legacy là một bộ phim đột phá về mặt âm nhạc, quay phim, hiệu ứng đặc biệt, và tôi rất mừng vì cuối cùng nó cũng được cả giới phê bình lẫn người hâm mộ đánh giá lại. Chúng ta đã quá bất công với bộ phim này vào năm 2010.", "Nam 1 (id_0001)"],
    ["Từ nhiều nguồn tài liệu lịch sử, có thể thấy nuôi con theo phong cách Do Thái không chỉ tốt cho đứa trẻ mà còn tốt cho cả các bậc cha mẹ.", "Nữ 1 (id_0002)"],
    ["Các bác sĩ đang nghiên cứu một loại vaccine mới chống lại virus cúm mùa. Thí nghiệm lâm sàng cho thấy phản ứng miễn dịch mạnh mẽ và ít tác dụng phụ, mở ra hy vọng phòng chống dịch bệnh hiệu quả hơn trong tương lai.", "Nam 2 (id_0003)"],
]

custom_css = """
.gradio-container { max-width: 1000px !important; margin: 0 auto !important; padding: 20px !important; }
.contain { max-width: 1000px !important; margin: 0 auto !important; }
#warning { background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 10px; margin: 10px 0; }
#info { background-color: #d1ecf1; border: 1px solid #17a2b8; border-radius: 5px; padding: 10px; margin: 10px 0; }
"""

with gr.Blocks(title="VieNeu-TTS Local", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ VieNeu-TTS: Vietnamese Text-to-Speech (Local Version)

    Hệ thống tổng hợp tiếng nói tiếng Việt được **finetune từ NeuTTS-Air**.

    Tác giả: [Phạm Nguyễn Ngọc Bảo](https://github.com/pnnbao97)  
    Model: [VieNeu-TTS](https://huggingface.co/pnnbao-ump/VieNeu-TTS)  
    Code: [GitHub](https://github.com/pnnbao97/VieNeu-TTS)
    """)
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="📝 Văn bản đầu vào (tối đa 250 ký tự)",
                placeholder="Nhập văn bản tiếng Việt...",
                lines=4,
                max_lines=6,
                value="Legacy là một bộ phim đột phá về mặt âm nhạc, quay phim, hiệu ứng đặc biệt, và tôi rất mừng vì cuối cùng nó cũng được cả giới phê bình lẫn người hâm mộ đánh giá lại. Chúng ta đã quá bất công với bộ phim này vào năm 2010."
            )
            char_count = gr.Markdown("209 / 250 ký tự")

            voice_select = gr.Radio(
                choices=list(VOICE_SAMPLES.keys()),
                label="🎤 Chọn giọng mẫu",
                value="Nam 1 (id_0001)",
                info="Giọng lẻ: Nam | Giọng chẵn: Nữ"
            )

            with gr.Accordion("🎨 Hoặc sử dụng giọng tùy chỉnh", open=False):
                gr.Markdown("""
                **Hướng dẫn:**
                - Upload file audio (.wav) và nhập nội dung text chính xác tương ứng
                - **Lưu ý:** Chất lượng có thể không tốt bằng các giọng mẫu
                """)
                custom_audio = gr.Audio(label="File audio mẫu", type="filepath")
                custom_text = gr.Textbox(label="Nội dung của audio mẫu", placeholder="Nhập chính xác nội dung...", lines=2)

            submit_btn = gr.Button("🎵 Tổng hợp giọng nói", variant="primary", size="lg")

            # Checkbox được đặt DƯỚI nút
            use_normalizer = gr.Checkbox(label="🧹 Bật chuẩn hoá văn bản", value=False)

        with gr.Column():
            # Audio tự phát
            audio_output = gr.Audio(label="🔊 Kết quả", autoplay=True)
            # "Sau chuẩn hoá" chuyển sang cột phải, đặt TRÊN khối trạng thái
            norm_output = gr.Textbox(label="🔧 Sau chuẩn hoá (chỉ hiện khi bật)", interactive=False, lines=4)
            status_output = gr.Textbox(label="📊 Trạng thái", interactive=False)

    gr.Markdown("### 💡 Ví dụ nhanh")
    gr.Examples(
        examples=examples,
        inputs=[text_input, voice_select],
        outputs=[audio_output, status_output, norm_output],
        fn=synthesize_speech,
        cache_examples=False
    )

    def update_char_count(text):
        count = len(text) if text else 0
        color = "red" if count > 250 else "green"
        return f"<span style='color: {color}'>{count} / 250 ký tự</span>"

    # cập nhật preview normalize theo text + checkbox
    text_input.change(
        fn=preview_normalize,
        inputs=[text_input, use_normalizer],
        outputs=[norm_output],
    )
    use_normalizer.change(
        fn=preview_normalize,
        inputs=[text_input, use_normalizer],
        outputs=[norm_output],
    )

    text_input.change(fn=update_char_count, inputs=[text_input], outputs=[char_count])

    submit_btn.click(
        fn=synthesize_speech,
        inputs=[text_input, voice_select, custom_audio, custom_text, use_normalizer],
        outputs=[audio_output, status_output, norm_output],
    )

if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, show_error=True)
