"""Process a list of (match_key, local video) pairs and store results (used for validation)."""
import sys
import traceback

sys.path.insert(0, ".")
from autoscout import db  # noqa: E402
from autoscout.pipeline import RESULTS_DIR, Options, process_match  # noqa: E402
from autoscout.vision.detect import default_detector  # noqa: E402

det = default_detector()
con = db.connect()
for arg in sys.argv[1:]:
    mk, video = arg.split("=")
    try:
        r = process_match(mk, Options(debug_video=True), video_path=video, detector=det)
        db.save_result(con, r, str(RESULTS_DIR / f"{mk}.json"))
    except Exception:
        traceback.print_exc()
