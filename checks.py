"""
checks.py  —  Dataset QA Tool V2.3
Universal YOLO layout detection — handles both structures automatically.

LAYOUT A  (standard YOLO / Ultralytics):
    dataset/
        images/
            train/   val/   test/   <any split>/
        labels/
            train/   val/   test/   <any split>/

LAYOUT B  (Roboflow YOLOv8 export):
    dataset/
        train/
            images/
            labels/
        valid/          ← also accepts "val"
            images/
            labels/
        test/
            images/
            labels/

detect_layout() sniffs which layout is present.
All check functions work identically regardless of layout — they receive
resolved (img_dir, lbl_dir) pairs per split from resolve_split_dirs().
"""

import os
import hashlib
from PIL import Image


# ── Supported image extensions ─────────────────────────────────────────────
IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff", ".webp",
)

# Layout constants
LAYOUT_A = "standard"   # images/<split>/  +  labels/<split>/
LAYOUT_B = "roboflow"   # <split>/images/  +  <split>/labels/
LAYOUT_UNKNOWN = "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  Layout detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_layout(dataset_path: str) -> str:
    """
    Sniff which directory layout the dataset uses.

    Layout A: dataset/images/ exists as a top-level folder.
    Layout B: dataset/<split>/images/ exists (split at root level).
    Returns LAYOUT_A, LAYOUT_B, or LAYOUT_UNKNOWN.
    """
    # Layout A: top-level images/ directory
    if os.path.isdir(os.path.join(dataset_path, "images")):
        return LAYOUT_A

    # Layout B: any root subfolder that itself contains images/ + labels/
    for entry in os.listdir(dataset_path):
        entry_path = os.path.join(dataset_path, entry)
        if not os.path.isdir(entry_path):
            continue
        has_images = os.path.isdir(os.path.join(entry_path, "images"))
        has_labels = os.path.isdir(os.path.join(entry_path, "labels"))
        if has_images or has_labels:
            return LAYOUT_B

    return LAYOUT_UNKNOWN


def get_splits(dataset_path: str) -> list[str]:
    """
    Return sorted list of split names present in the dataset,
    regardless of layout.

    Layout A → subdirs of  dataset/images/
    Layout B → root subdirs that contain images/ or labels/
    """
    layout = detect_layout(dataset_path)

    if layout == LAYOUT_A:
        images_root = os.path.join(dataset_path, "images")
        return sorted(
            d for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        )

    if layout == LAYOUT_B:
        splits = []
        for entry in sorted(os.listdir(dataset_path)):
            entry_path = os.path.join(dataset_path, entry)
            if not os.path.isdir(entry_path):
                continue
            if (os.path.isdir(os.path.join(entry_path, "images")) or
                    os.path.isdir(os.path.join(entry_path, "labels"))):
                splits.append(entry)
        return splits

    return []


def resolve_split_dirs(dataset_path: str, split: str) -> tuple[str, str]:
    """
    Return (img_dir, lbl_dir) for a given split, adapting to whichever
    layout is in use.

    Layout A:  (dataset/images/<split>,  dataset/labels/<split>)
    Layout B:  (dataset/<split>/images,  dataset/<split>/labels)
    """
    layout = detect_layout(dataset_path)

    if layout == LAYOUT_A:
        return (
            os.path.join(dataset_path, "images", split),
            os.path.join(dataset_path, "labels", split),
        )
    else:  # LAYOUT_B or fallback
        return (
            os.path.join(dataset_path, split, "images"),
            os.path.join(dataset_path, split, "labels"),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Structure validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset_structure(
    dataset_path: str,
) -> tuple[bool, list[str], list[str]]:
    """
    Validate dataset structure without assuming any specific layout or
    split names.  Accepts both Layout A and Layout B automatically.

    Returns:
        (is_valid, found_splits, problems)

        is_valid     — True when structure is usable
        found_splits — list of discovered split names
        problems     — human-readable list of problems (empty when valid)
    """
    problems: list[str] = []
    layout = detect_layout(dataset_path)

    if layout == LAYOUT_UNKNOWN:
        problems.append(
            "Could not detect a valid YOLO dataset layout.\n\n"
            "Expected either:\n"
            "  Layout A (standard):   dataset/images/<split>/  +  dataset/labels/<split>/\n"
            "  Layout B (Roboflow):   dataset/<split>/images/  +  dataset/<split>/labels/"
        )
        return (False, [], problems)

    splits = get_splits(dataset_path)

    if not splits:
        problems.append("No split folders found (e.g. train, valid, test).")
        return (False, [], problems)

    # Verify each split has both images and labels sub-directories
    for split in splits:
        img_dir, lbl_dir = resolve_split_dirs(dataset_path, split)

        if not os.path.isdir(img_dir):
            problems.append(f"Split '{split}': images folder not found at {img_dir}")
        if not os.path.isdir(lbl_dir):
            problems.append(f"Split '{split}': labels folder not found at {lbl_dir}")

    return (len(problems) == 0, splits, problems)


# ══════════════════════════════════════════════════════════════════════════════
#  Check functions  (all layout-agnostic via resolve_split_dirs)
# ══════════════════════════════════════════════════════════════════════════════

def get_image_label_counts(dataset_path: str, splits: list[str]) -> dict:
    """
    Return per-split and total image/label file counts.
    Keys: "<split>_images", "<split>_labels", "total_images", "total_labels"
    """
    stats: dict[str, int] = {}
    total_images = 0
    total_labels = 0

    for split in splits:
        img_dir, lbl_dir = resolve_split_dirs(dataset_path, split)

        img_count = len([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]) if os.path.isdir(img_dir) else 0

        lbl_count = len([
            f for f in os.listdir(lbl_dir)
            if f.endswith(".txt")
        ]) if os.path.isdir(lbl_dir) else 0

        stats[f"{split}_images"] = img_count
        stats[f"{split}_labels"] = lbl_count
        total_images += img_count
        total_labels += lbl_count

    stats["total_images"] = total_images
    stats["total_labels"] = total_labels
    return stats


def check_missing_pairs(
    dataset_path: str, splits: list[str]
) -> tuple[int, int, list[str], list[str]]:
    """
    Detect images without matching labels and labels without matching images.
    Returns: (missing_labels_count, missing_images_count, ml_files, mi_files)
    """
    missing_labels: list[str] = []
    missing_images: list[str] = []

    for split in splits:
        img_dir, lbl_dir = resolve_split_dirs(dataset_path, split)

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        imgs = {
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        }
        lbls = {
            os.path.splitext(f)[0]
            for f in os.listdir(lbl_dir)
            if f.endswith(".txt")
        }

        for stem in imgs - lbls:
            missing_labels.append(f"{split}/{stem}")
        for stem in lbls - imgs:
            missing_images.append(f"{split}/{stem}")

    return (len(missing_labels), len(missing_images), missing_labels, missing_images)


def check_corrupt_images(
    dataset_path: str, splits: list[str]
) -> tuple[int, list[str]]:
    """
    Verify every image with Pillow. Files that fail are reported as corrupt.
    Returns: (corrupt_count, corrupt_files)
    """
    corrupt_files: list[str] = []

    for split in splits:
        img_dir, _ = resolve_split_dirs(dataset_path, split)
        if not os.path.isdir(img_dir):
            continue

        for f in os.listdir(img_dir):
            if not f.lower().endswith(IMAGE_EXTENSIONS):
                continue
            path = os.path.join(img_dir, f)
            try:
                img = Image.open(path)
                img.verify()
            except Exception:
                corrupt_files.append(f"{split}/{f}")

    return (len(corrupt_files), corrupt_files)


def check_duplicates(
    dataset_path: str, splits: list[str]
) -> tuple[int, list[str]]:
    """
    Detect duplicate images via MD5 hash across all splits.
    Returns: (duplicate_count, duplicate_files)
    """
    hashes: dict[str, str] = {}
    duplicate_files: list[str] = []

    for split in splits:
        img_dir, _ = resolve_split_dirs(dataset_path, split)
        if not os.path.isdir(img_dir):
            continue

        for f in os.listdir(img_dir):
            if not f.lower().endswith(IMAGE_EXTENSIONS):
                continue
            path = os.path.join(img_dir, f)
            with open(path, "rb") as fh:
                file_hash = hashlib.md5(fh.read()).hexdigest()

            if file_hash in hashes:
                duplicate_files.append(f"{split}/{f}")
            else:
                hashes[file_hash] = path

    return (len(duplicate_files), duplicate_files)


def validate_labels(
    dataset_path: str, splits: list[str]
) -> tuple[int, int, list[str], list[str]]:
    """
    Validate every .txt label file: empty, wrong field count, out-of-range coords.
    Returns: (invalid_count, empty_count, invalid_files, empty_files)
    """
    invalid_files: list[str] = []
    empty_files: list[str] = []

    for split in splits:
        _, lbl_dir = resolve_split_dirs(dataset_path, split)
        if not os.path.isdir(lbl_dir):
            continue

        for f in os.listdir(lbl_dir):
            if not f.endswith(".txt"):
                continue
            path = os.path.join(lbl_dir, f)

            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            if not lines:
                empty_files.append(f"{split}/{f}")
                continue

            bad = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    bad = True
                    break
                try:
                    _cls, x, y, w, h = map(float, parts)
                    if not (0 <= x <= 1 and 0 <= y <= 1
                            and 0 <= w <= 1 and 0 <= h <= 1):
                        bad = True
                        break
                except ValueError:
                    bad = True
                    break

            if bad:
                invalid_files.append(f"{split}/{f}")

    return (len(invalid_files), len(empty_files), invalid_files, empty_files)

# ══════════════════════════════════════════════════════════════════════════════
#  Auto Fix Execution Logic (V3.0)
# ══════════════════════════════════════════════════════════════════════════════

def execute_auto_fixes(
    dataset_path: str,
    issue_details: dict[str, list[str]],
    fix_invalid: bool,
    delete_invalid_images: bool,
    log_callback
) -> dict[str, int]:
    """
    Executes selected dataset fixes using layout-agnostic absolute paths.
    Invokes log_callback(message, tag) for real-time reporting.
    Returns counts of deleted items.
    """
    deleted_counts = {
        "invalid_labels": 0,
        "invalid_images": 0,
        "duplicate_images": 0,
        "empty_labels": 0
    }
    


    # Helper to split "split/filename.ext" safely into absolute paths
    def get_abs_paths(file_entry: str, is_label: bool = False) -> tuple[str, str, str]:
        parts = file_entry.split("/", 1)
        if len(parts) != 2:
            return "", "", ""
        split, filename = parts
        img_dir, lbl_dir = resolve_split_dirs(dataset_path, split)
        target_dir = lbl_dir if is_label else img_dir
        return os.path.abspath(os.path.join(target_dir, filename)), split, os.path.splitext(filename)[0]

    # 1. Fix Invalid Labels & Corresponding Images
    if fix_invalid and issue_details.get("Invalid Labels"):
        for entry in issue_details["Invalid Labels"]:
            lbl_path, split, stem = get_abs_paths(entry, is_label=True)
            if lbl_path and os.path.isfile(lbl_path):
                try:
                    os.remove(lbl_path)
                    log_callback(f"Deleted invalid label: {entry}", "err")
                    deleted_counts["invalid_labels"] += 1
                except Exception as e:
                    log_callback(f"Failed to delete label {lbl_path}: {e}", "warn")

            # Hunt and delete corresponding image across all valid extensions
            if delete_invalid_images:

                img_dir, _ = resolve_split_dirs(dataset_path, split)

                if os.path.isdir(img_dir):

                    for ext in IMAGE_EXTENSIONS:

                        img_path = os.path.abspath(
                            os.path.join(img_dir, f"{stem}{ext}")
                        )

                        if os.path.isfile(img_path):

                            try:
                                os.remove(img_path)

                                log_callback(
                                    f"Deleted matching image: {split}/{stem}{ext}",
                                    "err"
                                )

                                deleted_counts["invalid_images"] += 1

                            except Exception as e:

                                log_callback(
                                    f"Failed to delete image {img_path}: {e}",
                                    "warn"
                                )

        return deleted_counts

                