import logging
import sys
import os
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Asia/Bangkok")
except Exception:
    _TZ = None  # fallback UTC

_LOGGER_INITIALIZED = False

class _DropPhonemizerSwitches(logging.Filter):
    """Filter bỏ các cảnh báo lặt vặt từ phonemizer/espeak."""
    PREFIXES = ("phonemizer", "espeak", "espeakng")
    PHRASES = (
        "language switches",
        "extra phones may appear",
        "words count mismatch",
    )
    def filter(self, record: logging.LogRecord) -> bool:
        name = getattr(record, "name", "") or ""
        # khớp "phonemizer" hoặc "phonemizer.*" (và espeak*)
        if any(name == p or name.startswith(p + ".") for p in self.PREFIXES):
            msg = str(record.getMessage()).lower()
            if any(ph in msg for ph in self.PHRASES):
                return False
        # nén cả warnings.warn được chuyển vào logger "py.warnings"
        if name == "py.warnings":
            msg = str(record.getMessage()).lower()
            if any(ph in msg for ph in self.PHRASES):
                return False
        return True

def _apply_quiet_third_party():
    """Giảm ồn: đặt level và tắt propagate cho logger con ồn ào."""
    for name in (
        "phonemizer",
        "phonemizer.backend",
        "phonemizer.phonemize",
        "espeak",
        "espeakng",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)   # chỉ ERROR trở lên
        lg.propagate = False         # không đẩy lên root
        # dọn handler nếu lib tự gắn
        try:
            for h in list(lg.handlers):
                lg.removeHandler(h)
        except Exception:
            pass

    # Bắt warnings.warn → logging và nén nó xuống
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.ERROR)

def setup_logging(
    run_name: str = "exp",
    to_file: bool = True,
    log_dir: str = "logs",
    level: str | int = "INFO",
    force: bool = False
):
    """
    Khởi tạo logging cho toàn bộ app.
    """
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.DEBUG)

    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # ---- handlers
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = None
    if to_file:
        ts = datetime.now(_TZ).strftime("%Y_%m_%d_%H_%M_%S") if _TZ else datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"vieneu_log_{ts}.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # ---- filter (gắn cả root + từng handler để chắc chắn)
    drop_filter = _DropPhonemizerSwitches()
    root.addFilter(drop_filter)
    ch.addFilter(drop_filter)
    if fh is not None:
        fh.addFilter(drop_filter)

    # ---- quiet noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _apply_quiet_third_party()

    # banner
    root.info("Logging initialized for run: %s", run_name)
    if fh is not None:
        root.info("File logging -> %s", log_path)
    else:
        root.info("Logging initialized for run: %s (console only)", run_name)

    _LOGGER_INITIALIZED = True
    logging.getLogger().propagate = True

def get_logger(name: str):
    return logging.getLogger(name)
