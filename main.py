import os
import time
import soundfile as sf
import torch

from utils.logging import setup_logging, get_logger
from vieneutts import VieNeuTTS

# Hiệu năng
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

setup_logging(run_name="vieneu-main", to_file=True, log_dir="logs", level="INFO")
log = get_logger("app.main")

def _gpu_mem_log(note: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        resvd = torch.cuda.memory_reserved() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        log.info("GPU VRAM%s alloc=%.1fMB,resvd=%.1fMB,peak=%.1fMB",
                 f" [{note}]" if note else "", alloc, resvd, peak)

input_texts = [
    "Các khóa học trực tuyến đang giúp học sinh tiếp cận kiến thức mọi lúc mọi nơi. Giáo viên sử dụng video, bài tập tương tác và thảo luận trực tuyến để nâng cao hiệu quả học tập.",
    "Các nghiên cứu về bệnh Alzheimer cho thấy tác dụng tích cực của các bài tập trí não và chế độ dinh dưỡng lành mạnh, giúp giảm tốc độ suy giảm trí nhớ ở người cao tuổi.",
    "Một tiểu thuyết trinh thám hiện đại dẫn dắt độc giả qua những tình tiết phức tạp, bí ẩn, kết hợp yếu tố tâm lý sâu sắc khiến người đọc luôn hồi hộp theo dõi diễn biến câu chuyện.",
    "Các nhà khoa học nghiên cứu gen người phát hiện những đột biến mới liên quan đến bệnh di truyền. Điều này giúp nâng cao khả năng chẩn đoán và điều trị.",
]

output_dir = "./output_audio"
os.makedirs(output_dir, exist_ok=True)

def main(backbone="pnnbao-ump/VieNeu-TTS", codec="neuphonic/neucodec"):
    # Chọn sample tham chiếu
    ref_audio_path = "./sample/id_0004.wav"  # Nữ 2
    ref_text_path = "./sample/id_0004.txt"

    if not os.path.exists(ref_audio_path) or not os.path.exists(ref_text_path):
        log.error("Thiếu reference audio/text.")
        return

    with open(ref_text_path, "r", encoding="utf-8") as f:
        ref_text = f.read()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Thiết bị: %s", device.upper())

    tts = VieNeuTTS(
        backbone_repo=backbone,
        backbone_device=device,
        codec_repo=codec,
        codec_device=device
    )
    _gpu_mem_log("sau load model")

    log.info("Encoding reference audio: %s", ref_audio_path)
    t0 = time.perf_counter()
    ref_codes = tts.encode_reference(ref_audio_path)
    log.info("encode_reference() %.1f ms", (time.perf_counter() - t0) * 1000)
    _gpu_mem_log("sau encode_reference")

    for i, text in enumerate(input_texts, 1):
        log.info("Generating example %d | len(text)=%d", i, len(text))
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
            wav = tts.infer(text, ref_codes, ref_text)
        dt = (time.perf_counter() - t0) * 1000
        out = os.path.join(output_dir, f"output_{i}.wav")
        sf.write(out, wav, 24000)
        log.info("Saved %s | infer=%.1f ms | dur=%.2fs", out, dt, wav.shape[-1] / 24000)
        _gpu_mem_log(f"after example {i}")

if __name__ == "__main__":
    main()
