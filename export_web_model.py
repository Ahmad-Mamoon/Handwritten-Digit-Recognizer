"""Export the trained model for the website (web/model/), e.g. to deploy it on GitHub Pages.

    python export_web_model.py

train.py already does this after training; run it yourself only to re-export an existing model.
"""

import argparse
import sys
from pathlib import Path

from digit_recognizer.config import MODEL_PATH, WEB_MODEL_DIR
from digit_recognizer.export import export_web_model
from digit_recognizer.model import ModelNotFoundError


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the trained model for in-browser use.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help=f"trained Keras model (default: {MODEL_PATH})")
    parser.add_argument("--output", type=Path, default=WEB_MODEL_DIR, help=f"output folder (default: {WEB_MODEL_DIR})")
    args = parser.parse_args()

    try:
        manifest = export_web_model(args.model, args.output)
    except ModelNotFoundError as exc:
        sys.exit(str(exc))
    print(f"Exported {args.model} -> {manifest.parent}")


if __name__ == "__main__":
    main()
