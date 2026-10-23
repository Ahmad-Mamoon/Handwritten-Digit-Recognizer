"""Run the pygame desktop app: python desktop.py"""

import sys

from digit_recognizer.model import ModelNotFoundError


def main() -> None:
    try:
        from digit_recognizer.desktop import run
    except ModuleNotFoundError as exc:
        if exc.name != "pygame":
            raise
        sys.exit("pygame isn't installed. Run: pip install -r requirements-desktop.txt")
    try:
        run()
    except ModelNotFoundError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
