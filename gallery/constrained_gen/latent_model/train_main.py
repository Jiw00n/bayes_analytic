from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    from config import build_config
    from train import train_main
else:
    from .config import build_config
    from .train import train_main


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            "CLI overrides were removed. Edit latent_param_model/config.py and run `python main.py`."
        )
    train_main(build_config())


if __name__ == "__main__":
    main()
