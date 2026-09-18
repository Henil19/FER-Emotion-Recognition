"""Validate the FER-2013 directory structure before training."""
from pathlib import Path

EXPECTED_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "fer2013"


def validate_split(split_dir):
    missing = []
    counts = {}
    for class_name in EXPECTED_CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            missing.append(class_name)
            counts[class_name] = 0
            continue
        counts[class_name] = sum(1 for path in class_dir.iterdir() if path.is_file())
    return missing, counts


def main():
    print(f"Checking dataset at: {DATA_DIR}")
    if not DATA_DIR.is_dir():
        raise SystemExit(
            "FER-2013 dataset directory not found. Create fer2013/train and "
            "fer2013/test with the seven emotion folders."
        )

    failed = False
    for split in ("train", "test"):
        split_dir = DATA_DIR / split
        missing, counts = validate_split(split_dir)
        print(f"\n{split.upper()} split:")
        for class_name in EXPECTED_CLASSES:
            print(f"  {class_name:8s}: {counts[class_name]:6d} images")
        if missing:
            print(f"  Missing folders: {', '.join(missing)}")
            failed = True

    if failed:
        raise SystemExit("Dataset validation failed.")

    print("\nDataset structure looks valid. Ready for training.")


if __name__ == "__main__":
    main()
