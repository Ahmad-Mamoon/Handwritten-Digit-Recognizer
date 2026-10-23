"""Run the website (web/) locally, plus the JSON prediction API.

    python app.py                  # http://127.0.0.1:5000
    python app.py --host 0.0.0.0   # also reachable from phones/tablets on your network

The website itself needs no server (it's deployed to GitHub Pages as static files).
"""

import argparse
import os

from digit_recognizer.model import ModelNotFoundError
from digit_recognizer.server import create_app

# Module-level app so it also works with `flask --app app run` or a WSGI server (waitress, gunicorn).
app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the handwritten digit recognizer web app.")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="port to listen on (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="auto-reload on code changes and show the Flask debugger")
    args = parser.parse_args()

    # In debug mode the reloader serves requests from a child process, so only warm up there.
    if not args.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        classifier = app.extensions["digit_classifier"]
        try:
            classifier.warm_up()
            print(f" * Model loaded from {classifier.model_path}")
        except ModelNotFoundError as exc:
            print(f" * WARNING: {exc}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
