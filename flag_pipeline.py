import os, sys, yaml, shutil, hashlib, random, warnings
import cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import albumentations as A
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ─── Config ────────────────────────────────────────────────────────────────────
DATASET_ROOT   = Path("country-flags-2t33e-4")   # set via --dataset
PROCESSED_ROOT = Path("processed_dataset")
AUG_ROOT       = Path("augmented_dataset")
OUTPUTS_DIR    = Path("outputs")
SPLITS         = ["train", "valid", "test"]
RESIZE_W       = 640
RESIZE_H       = 640
AUG_FACTOR     = 2
SEED           = 42
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
api_key = os.getenv("ROBOFLOW_API_KEY")

random.seed(SEED)
np.random.seed(SEED)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_image_paths(split: str) -> list[Path]:
    img_dir = DATASET_ROOT / split / "images"
    return [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]


def parse_yolo_labels(label_path: Path):
    boxes, classes = [], []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    classes.append(int(parts[0]))
                    boxes.append([float(x) for x in parts[1:]])
    return boxes, classes


def save_yolo_labels(label_path: Path, boxes, classes):
    with open(label_path, "w") as f:
        for cls, box in zip(classes, boxes):
            f.write(f'{cls} {" ".join(f"{v:.6f}" for v in box)}\n')


def file_hash(path: Path, chunk: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def count_classes_in_dir(label_dir: Path) -> Counter:
    counts: Counter = Counter()
    for lf in Path(label_dir).glob("*.txt"):
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1
    return counts


# ─── Phase 1: Investigation ────────────────────────────────────────────────────
def phase1_investigation(class_names: list[str]) -> dict:
    total_classes = len(class_names)
    print("\n" + "="*60)
    print("PHASE 1 — DATA INVESTIGATION")
    print("="*60)

    # 1.1 Image counts
    split_counts = {s: len(get_image_paths(s)) for s in SPLITS}
    total = sum(split_counts.values())
    print(f"\n{'Split':<12} {'Count':>8}  {'%':>6}")
    print("─" * 30)
    for s, c in split_counts.items():
        print(f"{s:<12} {c:>8}  {c/total*100:>5.1f}%")
    print(f"{'TOTAL':<12} {total:>8}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(split_counts.keys(), split_counts.values(),
           color=["#4C72B0", "#55A868", "#C44E52"], edgecolor="white")
    ax.set_title("Images per Split", fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "split_counts.png", dpi=150)
    plt.close()

    # 1.2 Class occurrences
    all_counts: Counter = Counter()
    for s in SPLITS:
        label_dir = DATASET_ROOT / s / "labels"
        for lf in label_dir.glob("*.txt"):
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        all_counts[int(parts[0])] += 1

    represented = set(all_counts.keys())
    missing     = set(range(total_classes)) - represented
    rare        = {cid: cnt for cid, cnt in all_counts.items() if cnt < 5}

    print(f"\nClasses represented : {len(represented)} / {total_classes}")
    print(f"Missing classes     : {len(missing)}")
    if missing:
        print("  →", [class_names[i] for i in sorted(missing)])
    print(f"Rare (<5 samples)   : {len(rare)}")

    # 1.3 Class imbalance chart
    counts_df = pd.DataFrame([
        {"class_id": cid, "class_name": class_names[cid], "count": cnt}
        for cid, cnt in sorted(all_counts.items())
    ]).sort_values("count", ascending=False)

    max_c = counts_df["count"].max()
    min_c = counts_df["count"].min()
    print(f"\nImbalance ratio     : {max_c / max(min_c,1):.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    palette = sns.color_palette("viridis", len(counts_df))
    axes[0].bar(range(len(counts_df)), counts_df["count"].values, color=palette)
    axes[0].axhline(counts_df["count"].mean(), color="red", linestyle="--",
                    label=f"Mean={counts_df['count'].mean():.0f}")
    axes[0].set_title("Class Distribution", fontweight="bold")
    axes[0].legend()

    top20 = counts_df.head(20); bot20 = counts_df.tail(20)
    combined = pd.concat([top20, bot20])
    axes[1].barh(range(len(combined)), combined["count"].values,
                 color=["#2ecc71"]*20 + ["#e74c3c"]*20)
    axes[1].set_yticks(range(len(combined)))
    axes[1].set_yticklabels(combined["class_name"].values, fontsize=7)
    axes[1].set_title("Top 20 vs Bottom 20", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # 1.4 Resolution analysis
    resolutions = []
    for split in SPLITS:
        for img_path in tqdm(get_image_paths(split), desc=f"  Res scan {split}"):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    resolutions.append({"split": split, "width": w, "height": h})
            except Exception:
                pass
    res_df = pd.DataFrame(resolutions)
    print(f"\nMin  : {res_df['width'].min()}x{res_df['height'].min()}")
    print(f"Max  : {res_df['width'].max()}x{res_df['height'].max()}")
    print(f"Mean : {res_df['width'].mean():.0f}x{res_df['height'].mean():.0f}")
    most_common = res_df.groupby(["width","height"]).size().idxmax()
    print(f"Most common: {most_common}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(res_df["width"],  bins=30, color="steelblue",      edgecolor="white")
    axes[0].set_title("Width Distribution")
    axes[1].hist(res_df["height"], bins=30, color="coral",          edgecolor="white")
    axes[1].set_title("Height Distribution")
    sc = res_df.groupby(["width","height"]).size().reset_index(name="count") \
               .sort_values("count", ascending=False).head(15)
    axes[2].barh([f"{r.width}x{r.height}" for _, r in sc.iterrows()][::-1],
                 sc["count"].values[::-1], color="mediumseagreen")
    axes[2].set_title("Top 15 Resolutions")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "resolution_analysis.png", dpi=150)
    plt.close()

    # 1.5 BBox analysis
    bbox_stats = []
    for split in SPLITS:
        label_dir = DATASET_ROOT / split / "labels"
        for lf in tqdm(list(label_dir.glob("*.txt")), desc=f"  BBox {split}"):
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, bw, bh = parts
                        bbox_stats.append({
                            "bw": float(bw), "bh": float(bh),
                            "area_pct": float(bw)*float(bh)*100
                        })
    bb_df = pd.DataFrame(bbox_stats)
    print(f"\nSmall objects (area<1%): {(bb_df['area_pct']<1).sum()} / {len(bb_df)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(bb_df["bw"],       bins=40, color="#5B9BD5", edgecolor="white")
    axes[0].set_title("BBox Width (norm.)")
    axes[1].hist(bb_df["bh"],       bins=40, color="#ED7D31", edgecolor="white")
    axes[1].set_title("BBox Height (norm.)")
    axes[2].hist(bb_df["area_pct"], bins=40, color="#70AD47", edgecolor="white")
    axes[2].axvline(1, color="red", linestyle="--", label="1% threshold")
    axes[2].set_title("BBox Area %")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "bbox_analysis.png", dpi=150)
    plt.close()

    print("\n✅ Phase 1 complete. Charts saved to outputs/")
    return {"all_counts": all_counts, "total_classes": total_classes}


# ─── Phase 2: Cleaning ─────────────────────────────────────────────────────────
def phase2_cleaning(total_classes: int):
    print("\n" + "="*60)
    print("PHASE 2 — DATA CLEANING")
    print("="*60)

    # Corrupted
    corrupted = []
    for split in SPLITS:
        for img_path in tqdm(get_image_paths(split), desc=f"  Corrupt check {split}"):
            try:
                img = Image.open(img_path); img.verify()
            except Exception as e:
                corrupted.append(img_path)
    print(f"\nCorrupted: {len(corrupted)}")
    for p in corrupted:
        p.unlink(missing_ok=True)
        (p.parent.parent / "labels" / (p.stem + ".txt")).unlink(missing_ok=True)

    # Duplicates
    seen, duplicates = {}, []
    for split in SPLITS:
        for img_path in tqdm(get_image_paths(split), desc=f"  Dup check {split}"):
            h = file_hash(img_path)
            if h in seen:
                duplicates.append(img_path)
            else:
                seen[h] = img_path
    print(f"Duplicates: {len(duplicates)}")
    for p in duplicates:
        p.unlink(missing_ok=True)
        (p.parent.parent / "labels" / (p.stem + ".txt")).unlink(missing_ok=True)

    # Annotation errors
    issues, fixed = [], 0
    for split in SPLITS:
        label_dir = DATASET_ROOT / split / "labels"
        for lf in tqdm(list(label_dir.glob("*.txt")), desc=f"  Label fix {split}"):
            valid_lines = []
            with open(lf) as f:
                original = f.readlines()
            for line in original:
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append({"file": str(lf), "issue": "wrong fields"}); continue
                cls = int(parts[0])
                if cls < 0 or cls >= total_classes:
                    issues.append({"file": str(lf), "issue": f"invalid class {cls}"}); continue
                cx, cy, bw, bh = [float(x) for x in parts[1:]]
                if bw <= 0 or bh <= 0:
                    issues.append({"file": str(lf), "issue": "zero box"}); continue
                cx = max(0., min(1., cx)); cy = max(0., min(1., cy))
                bw = max(0.001, min(1., bw)); bh = max(0.001, min(1., bh))
                valid_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            if len(valid_lines) != len(original):
                fixed += 1
                with open(lf, "w") as f:
                    f.writelines(valid_lines)

    print(f"Annotation issues: {len(issues)} | Label files fixed: {fixed}")
    print("✅ Phase 2 complete.")


# ─── Phase 3: Preprocessing ────────────────────────────────────────────────────
def phase3_preprocessing():
    print("\n" + "="*60)
    print("PHASE 3 — PREPROCESSING  (resize to 640×640)")
    print("="*60)
    ok = fail = 0
    for split in SPLITS:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"
        dst_img = PROCESSED_ROOT / split / "images"
        dst_lbl = PROCESSED_ROOT / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img_path in tqdm(get_image_paths(split), desc=f"  Process {split}"):
            img = cv2.imread(str(img_path))
            if img is None:
                fail += 1; continue
            img_resized = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(dst_img / img_path.name), img_resized)
            lbl_src = lbl_dir / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy(lbl_src, dst_lbl / lbl_src.name)
            ok += 1
    shutil.copy(DATASET_ROOT / "data.yaml", PROCESSED_ROOT / "data.yaml")
    print(f"OK: {ok}  |  Failed: {fail}")
    print("✅ Phase 3 complete.")


