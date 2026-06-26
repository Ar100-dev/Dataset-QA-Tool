"""
analyzer.py  —  Dataset QA Tool V2.3
YAML class-name loading and class distribution, layout-agnostic.

Changes from V2.2:
    • load_class_names() searches for data.yaml in multiple locations:
        1. dataset/data.yaml          (Layout A standard)
        2. dataset/<split>/data.yaml  (fallback — Roboflow sometimes puts it inside)
    • class_distribution() uses resolve_split_dirs() from checks so it reads
      labels from the correct path regardless of Layout A or Layout B.
"""

import os
from collections import Counter
from checks import detect_layout, get_splits, resolve_split_dirs


# ── YAML class-name loader ──────────────────────────────────────────────────
def load_class_names(dataset_path: str) -> dict[int, str]:
    """
    Parse the 'names:' block from data.yaml.

    Searches in order:
      1. <dataset>/data.yaml           ← standard location
      2. <dataset>/<first_split>/data.yaml  ← Roboflow sometimes puts it here

    Returns {class_id: class_name}. Empty dict if not found.
    """
    candidates = [os.path.join(dataset_path, "data.yaml")]

    # Also check inside each split folder (Roboflow layout)
    for split in get_splits(dataset_path):
        candidates.append(os.path.join(dataset_path, split, "data.yaml"))

    yaml_path = None
    for c in candidates:
        if os.path.isfile(c):
            yaml_path = c
            break

    if not yaml_path:
        return {}

    class_names: dict[int, str] = {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        inside_names = False
        class_id = 0

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("names:"):
                inside_names = True
                # Handle inline list:  names: [car, truck, person]
                inline = stripped[len("names:"):].strip()
                if inline.startswith("["):
                    inline = inline.strip("[]")
                    for name in inline.split(","):
                        name = name.strip().strip("'\"")
                        if name:
                            class_names[class_id] = name
                            class_id += 1
                    inside_names = False
                continue

            if inside_names:
                # Stop at next top-level key (no indent, no dash)
                if stripped and not stripped.startswith("-") and not line.startswith(" "):
                    inside_names = False
                    continue
                if stripped.startswith("-"):
                    name = stripped.lstrip("-").strip().strip("'\"")
                    if name:
                        class_names[class_id] = name
                        class_id += 1

    except Exception as e:
        print(f"[analyzer] Failed to parse {yaml_path}: {e}")

    return class_names


# ── Class distribution counter ──────────────────────────────────────────────
def class_distribution(dataset_path: str) -> dict[str, int]:
    """
    Count annotation occurrences for each class across all splits,
    using layout-aware path resolution.

    Returns {class_name: count} using names from data.yaml when available,
    otherwise 'Class <id>'.
    """
    counts: Counter = Counter()
    splits = get_splits(dataset_path)

    for split in splits:
        _, lbl_dir = resolve_split_dirs(dataset_path, split)

        if not os.path.isdir(lbl_dir):
            continue

        for file in os.listdir(lbl_dir):
            if not file.endswith(".txt"):
                continue

            path = os.path.join(lbl_dir, file)
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls = int(float(parts[0]))
                        counts[cls] += 1
            except Exception:
                continue

    class_names = load_class_names(dataset_path)

    return {
        class_names.get(cls_id, f"Class {cls_id}"): count
        for cls_id, count in counts.items()
    }