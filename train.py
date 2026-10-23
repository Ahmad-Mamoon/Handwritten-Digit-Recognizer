"""Train the handwritten digit model on MNIST.

    python train.py               # up to 15 epochs; saves models/digit_model.keras and web/model/
    python train.py --epochs 3    # quicker, slightly less accurate
"""

import argparse
from pathlib import Path

from digit_recognizer.config import MODEL_PATH, WEB_MODEL_DIR
from digit_recognizer.export import export_web_model
from digit_recognizer.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the handwritten digit model on MNIST.")
    parser.add_argument("--epochs", type=int, default=15, help="maximum epochs; training stops early once accuracy plateaus (default: 15)")
    parser.add_argument("--batch-size", type=int, default=128, help="training batch size (default: 128)")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help=f"where to save the model (default: {MODEL_PATH})")
    parser.add_argument("--no-augment", action="store_true", help="skip data augmentation (faster, but worse on real handwriting)")
    parser.add_argument("--no-export", action="store_true", help=f"don't export the model for the website ({WEB_MODEL_DIR})")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible runs (default: 42)")
    args = parser.parse_args()

    metadata = train(
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        seed=args.seed,
    )
    print(f"\nTest accuracy: {metadata['test_accuracy']:.2%}")
    print(f"Model saved to {args.output}")
    if not args.no_export:
        export_web_model(args.output, WEB_MODEL_DIR)
        print(f"Website model exported to {WEB_MODEL_DIR}")


if __name__ == "__main__":
    main()
