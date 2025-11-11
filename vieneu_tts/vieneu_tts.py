from pathlib import Path
from typing import Generator
import time
import librosa
import numpy as np
import torch
from torch.backends.cuda import sdp_kernel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)


from neucodec import NeuCodec, DistillNeuCodec
from transformers import AutoTokenizer, AutoModelForCausalLM
from phonemizer.backend.espeak.espeak import EspeakWrapper
from phonemizer import phonemize
import platform
import re
import os
from functools import lru_cache

from utils.logging import get_logger
_log = get_logger("vieneu.core")

if platform.system() == "Windows":
    EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
elif platform.system() == "Linux":
    EspeakWrapper.set_library("/usr/lib/x86_64-linux-gnu/libespeak-ng.so")
elif platform.system() == "Darwin":
    espeak_lib = os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')
    if espeak_lib and os.path.exists(espeak_lib):
        EspeakWrapper.set_library(espeak_lib)
    elif os.path.exists("/opt/homebrew/lib/libespeak-ng.dylib"):
        EspeakWrapper.set_library("/opt/homebrew/lib/libespeak-ng.dylib")
    elif os.path.exists("/usr/local/lib/libespeak-ng.dylib"):
        EspeakWrapper.set_library("/usr/local/lib/libespeak-ng.dylib")
    elif os.path.exists("/opt/local/lib/libespeak-ng.dylib"):
        EspeakWrapper.set_library("/opt/local/lib/libespeak-ng.dylib")
    else:
        raise ValueError(
            "Không tìm thấy libespeak-ng.dylib. Cài bằng `brew install espeak` "
            "hoặc set PHONEMIZER_ESPEAK_LIBRARY=/path/to/libespeak-ng.dylib"
        )
else:
    raise ValueError(f"Unsupported platform: {platform.system()}")

def _linear_overlap_add(frames: list[np.ndarray], stride: int) -> np.ndarray:
    # original impl --> https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)
    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)
    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))
        out[..., offset: offset + frame_length] += weight * frame
        sum_weight[offset: offset + frame_length] += weight
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight

class VieNeuTTS:
    def __init__(
        self,
        backbone_repo="pnnbao-ump/VieNeu-TTS",
        codec_repo="neuphonic/neucodec",
    ):
        self.log = get_logger("vieneu.tts")
        # Determine the device to use
        if torch.cuda.is_available():
            self.backbone_device = "cuda"
        else:
            self.backbone_device = "cpu"
        self.codec_device = self.backbone_device
        self.backbone_repo = backbone_repo
        self.codec_repo = codec_repo

        # Const
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        # flags
        self._is_quantized_model = False
        self._is_onnx_codec = False

        # HF tokenizer
        self.tokenizer = None

        # Load models
        self._load_backbone()
        self._load_codec()

    def _gpu_mem_log(self, note: str = ""):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            resvd = torch.cuda.memory_reserved() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            self.log.info("GPU VRAM%s alloc=%.1fMB,resvd=%.1fMB,peak=%.1fMB",
                          f" [{note}]" if note else "", alloc, resvd, peak)

    def _to_int_list(self, x):
        import torch, numpy as np
        if isinstance(x, torch.Tensor):
            return [int(t.item()) for t in x.flatten()]
        if isinstance(x, np.ndarray):
            return [int(v) for v in x.flatten().tolist()]
        return [int(v) for v in x]  # list-like

    def _require_token(self, tokenizer, token_str: str) -> int:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer thiếu special token: {token_str} (trả về unk)")
        return int(tid)


    def _load_backbone(self):
        self.log.info("Loading backbone: %s on %s ...", self.backbone_repo, self.backbone_device)
        t0 = time.perf_counter()

        if self.backbone_repo.lower().endswith("gguf") or "gguf" in self.backbone_repo.lower():
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError("Cần `llama-cpp-python` để chạy GGUF") from e
            self.backbone = Llama.from_pretrained(
                repo_id=self.backbone_repo,
                filename="*.gguf",
                verbose=False,
                n_gpu_layers=-1 if self.backbone_device in ("cuda","gpu") else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if self.backbone_device in ("cuda","gpu") else False,
            )
            self._is_quantized_model = True
            self.log.info("Backbone GGUF đã nạp (quantized)")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path= self.backbone_repo,
                ).to(
                torch.device(self.backbone_device)
            )

            self.backbone.eval()  # tránh dropout/layernorm behaving như train
            self._validate_special_tokens()  # thêm hàm này ở dưới
            self.log.info("Backbone HF loaded → device=%s, dtype=%s",
                          self.backbone_device, getattr(self.backbone.dtype, "__str__", lambda: str(self.backbone.dtype))())
        self._gpu_mem_log("sau load backbone")
        self.log.info("Thời gian load backbone: %.1f ms", (time.perf_counter() - t0) * 1000)

    def _load_codec(self):
        self.log.info("Loading codec: %s on %s ...", self.codec_repo, self.codec_device)
        t0 = time.perf_counter()
        match self.codec_repo:
            case "neuphonic/neucodec":
                self.codec = NeuCodec.from_pretrained(self.codec_repo)
                self.codec.eval().to(self.codec_device)
            case "neuphonic/distill-neucodec":
                self.codec = DistillNeuCodec.from_pretrained(self.codec_repo)
                self.codec.eval().to(self.codec_device)
            case "neuphonic/neucodec-onnx-decoder":
                if self.codec_device != "cpu":
                    raise ValueError("Onnx decoder hiện chỉ chạy CPU.")
                try:
                    from neucodec import NeuCodecOnnxDecoder
                except ImportError as e:
                    raise ImportError("Thiếu onnxruntime / neucodec>=0.0.4") from e
                self.codec = NeuCodecOnnxDecoder.from_pretrained(self.codec_repo)
                self._is_onnx_codec = True
            case _:
                raise ValueError(f"Unsupported codec repository: {self.codec_repo}")
        self._gpu_mem_log("sau load codec")
        self.log.info("Thời gian load codec: %.1f ms", (time.perf_counter() - t0) * 1000)

    def infer(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> np.ndarray:
        """
        Perform inference to generate speech from text using the TTS model and reference audio.

        Args:
            text (str): Input text to be converted to speech.
            ref_codes (np.ndarray | torch.tensor): Encoded reference.
            ref_text (str): Reference text for reference audio. Defaults to None.
        Returns:
            np.ndarray: Generated speech waveform.
        """
        self.log.info("infer(): len(text)=%d", len(text))
        t0 = time.perf_counter()
        ref_codes_list = self._to_int_list(ref_codes)
        # Tạo chuỗi token đầu ra
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes_list, ref_text, text)
        else:
            prompt_ids = self._apply_chat_template(ref_codes_list, ref_text, text)
            self.log.info("prompt_len=%d / max_context=%d", len(prompt_ids), self.max_context)
            output_str = self._infer_torch(prompt_ids)

        # Giải mã speech tokens → wav
        wav = self._decode(output_str)
        dt = (time.perf_counter() - t0) * 1000
        self.log.info("infer() done: %.1f ms, len(wav)=%.2fs",
                      dt, wav.shape[-1] / float(self.sample_rate))
        return wav

    def infer_stream(self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str) -> Generator[np.ndarray, None, None]:
        if self._is_quantized_model:
            return self._infer_stream_ggml(ref_codes, ref_text, text)
        else:
            raise NotImplementedError("Streaming chưa hỗ trợ cho backend torch!")

    def encode_reference(self, ref_audio_path: str | Path):
        self.log.info("encode_reference(): %s", ref_audio_path)
        t0 = time.perf_counter()
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        with torch.inference_mode():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        self.log.info("encode_reference() xong: %.1f ms, codes=%s",
                      (time.perf_counter() - t0) * 1000, tuple(ref_codes.shape) if hasattr(ref_codes, "shape") else "unknown")
        return ref_codes

    def _decode(self, codes: str):
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]
        if len(speech_ids) == 0:
            raise ValueError("Không tìm thấy speech tokens hợp lệ trong output.")
        t0 = time.perf_counter()
        if self._is_onnx_codec:
            arr = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(arr)
        else:
            with torch.inference_mode():
                tens = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec.device)
                recon = self.codec.decode_code(tens).cpu().numpy()
        self.log.debug("_decode(): %.1f ms, n_tokens=%d", (time.perf_counter() - t0) * 1000, len(speech_ids))
        return recon[0, 0, :]

    # Cache phonemizer để tránh gọi eSpeak NG lặp lại
    @lru_cache(maxsize=4096)
    def _phonemize_cached(self, text: str) -> str:
        return phonemize(
            text,
            language="vi",
            backend="espeak",
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
        )


    def _to_phones(self, text: str) -> str:
        """Convert text to phonemes using phonemizer."""
        t0 = time.perf_counter()
        phones = self._phonemize_cached(text)
        
        # Handle both string and list returns
        if isinstance(phones, list):
            if len(phones) == 0:
                raise ValueError(f"Phonemization failed for text: {text}")
            res = phones[0]
        elif isinstance(phones, str):
            res = phones
        else:
            raise TypeError(f"Unexpected phonemize return type: {type(phones)}")
        self.log.debug("_to_phones(): %.1f ms, in_len=%d, out_len=%d",
                       (time.perf_counter() - t0) * 1000, len(text), len(res))
        return res

    def _validate_special_tokens(self):
        required = [
            "<|SPEECH_REPLACE|>",
            "<|SPEECH_GENERATION_START|>",
            "<|SPEECH_GENERATION_END|>",
            "<|TEXT_REPLACE|>",
            "<|TEXT_PROMPT_START|>",
            "<|TEXT_PROMPT_END|>",
        ]
        for tok in required:
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == self.tokenizer.unk_token_id:
                raise ValueError(f"Thiếu token đặc biệt trong tokenizer: {tok}")
        self.log.info("Special tokens OK")

    def _apply_chat_template(self, ref_codes: list[int], ref_text: str, input_text: str) -> list[int]:
        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)

        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1 :]
        )

        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)
        return ids

    def _infer_torch(self, prompt_ids: list[int]) -> str:
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(self.backbone.device)

        #  Logging token ids gây chậm → chỉ dùng khi debug
        for s in ["<|SPEECH_REPLACE|>", "<|SPEECH_GENERATION_START|>", "<|SPEECH_GENERATION_END|>",
          "<|TEXT_REPLACE|>", "<|TEXT_PROMPT_START|>", "<|TEXT_PROMPT_END|>"]:
            tid = self.tokenizer.convert_tokens_to_ids(s)
            _log.info("token %-28s -> %s", s, tid)

        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        if speech_end_id is None:
            speech_end_id = self.tokenizer.eos_token_id

        # Tính độ dài prompt và ngân sách new tokens an toàn
        prompt_len = int(prompt_tensor.shape[-1])
        # chừa 32 token “đệm” để tránh chạm trần
        room = max(0, self.max_context - prompt_len - 32)
        max_new = int(min(256, max(32, room)))  # 32..256

        # Nếu không còn chỗ → cắt đuôi prompt (cảnh báo)
        if max_new <= 0:
            keep = self.max_context - 96  # chừa room
            self.log.warning(
                "Prompt quá dài (len=%d ≥ %d). Cắt giữ %d token cuối.",
                prompt_len, self.max_context, keep
            )
            prompt_tensor = prompt_tensor[:, -keep:]
            prompt_len = int(prompt_tensor.shape[-1])
            room = max(0, self.max_context - prompt_len - 32)
            max_new = int(min(128, max(32, room)))  # sau khi cắt, vẫn giữ nhỏ

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                use_cache=True,
                min_new_tokens=50,
            )
        dt_ms = (time.perf_counter() - t0) * 1000
        input_length = prompt_tensor.shape[-1]
        new_len = int(out.shape[-1] - input_length)  # số LLM tokens mới

        # Tính tok/s cho LLM
        gen_s = max(dt_ms / 1000.0, 1e-6)
        llm_tps = new_len / gen_s

        # Lấy chuỗi phần sinh thêm để đếm speech tokens
        gen_token_ids = out[0, input_length:].detach().cpu().tolist()
        output_str = self.tokenizer.decode(gen_token_ids, add_special_tokens=False)
        speech_ids = re.findall(r"<\|speech_(\d+)\|>", output_str)
        speech_tps = (len(speech_ids) / gen_s) if speech_ids else 0.0

        self.log.info(
            "_infer_torch(): %.1f ms | new_llm=%d (%.2f tok/s) | new_speech=%d (%.2f tok/s)",
            dt_ms, new_len, llm_tps, len(speech_ids), speech_tps
        )

        return output_str


    def _infer_ggml(self, ref_codes: list[int], ref_text: str, input_text: str) -> str:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        t0 = time.perf_counter()
        out = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        text = out["choices"][0]["text"]

        # Nếu llama-cpp trả về usage/timings -> dùng để log LLM tok/s
        llm_tps = None
        try:
            usage = out.get("usage") or {}
            timings = out.get("timings") or {}
            comp = usage.get("completion_tokens")
            pred_ms = timings.get("predicted_ms") or timings.get("eval_duration_ms")
            if comp is not None and pred_ms:
                llm_tps = comp / max(pred_ms / 1000.0, 1e-6)
        except Exception:
            pass

        # Luôn log speech-tok/s theo chuỗi đã sinh
        speech_ids = re.findall(r"<\|speech_(\d+)\|>", text)
        speech_tps = (len(speech_ids) / max(dt_ms / 1000.0, 1e-6)) if speech_ids else 0.0

        if llm_tps is not None:
            self.log.info("_infer_ggml(): %.1f ms | new_speech=%d (%.2f tok/s) | LLM=%.2f tok/s",
                          dt_ms, len(speech_ids), speech_tps, llm_tps)
        else:
            self.log.info("_infer_ggml(): %.1f ms | new_speech=%d (%.2f tok/s)",
                          dt_ms, len(speech_ids), speech_tps)

        return text


    def _infer_stream_ggml(self, ref_codes: torch.Tensor, ref_text: str, input_text: str) -> Generator[np.ndarray, None, None]:
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)

        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=0.2,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True
        ):
            output_str = item["choices"][0]["text"]
            token_cache.append(output_str)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                tokens_start = max(n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0)
                tokens_end = (n_decoded_tokens + self.streaming_frames_per_chunk + self.streaming_lookforward + self.streaming_overlap_frames)
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = sample_start + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        remaining_tokens = len(token_cache) - n_decoded_tokens
        if len(token_cache) > n_decoded_tokens:
            tokens_start = max(len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens), 0)
            sample_start = (len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = recon[sample_start:]
            audio_cache.append(recon)
            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon
