"""Train semseg fcn_in (ImageNet backbone, vseg01), then evaluate on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.semseg.models import make_build

if __name__ == "__main__":
    main("train", task="semseg", version="fcn_in", build_model=make_build("fcn_in"))
