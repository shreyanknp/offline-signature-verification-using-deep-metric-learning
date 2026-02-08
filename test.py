# EDA for CEDAR Offline Signature Dataset (full_org + full_forg)
# - Image size/resolution distribution
# - Pixel intensity stats (grayscale)
# - Ink (foreground) density after Otsu binarization
# - Visual samples: intra-class variability + genuine vs forgery
#
# Requirements: pip install pandas numpy pillow matplotlib opencv-python

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# =========================
# 0) Paths + filename parse
# =========================
DATA_ROOT = Path(r"W:\SRH study\Case Study 2\Offline Signature Verification\Datasets\signatures")
ORG_DIR  = DATA_ROOT / "full_org"
FORG_DIR = DATA_ROOT / "full_forg"

PAT_ORG  = re.compile(r"^original_(\d+)_(\d+)\.png$", re.IGNORECASE)
PAT_FORG = re.compile(r"^forgeries_(\d+)_(\d+)\.png$", re.IGNORECASE)

def build_metadata():
    rows = []

    def scan(folder: Path, label: str, pat: re.Pattern):
        for fp in folder.glob("*.png"):
            m = pat.match(fp.name)
            if not m:
                rows.append({
                    "writer_id": None,
                    "sample_id": None,
                    "label": label,
                    "path": str(fp),
                    "filename_ok": False
                })
                continue
            rows.append({
                "writer_id": int(m.group(1)),
                "sample_id": int(m.group(2)),
                "label": label,            # "genuine" or "forgery"
                "path": str(fp),
                "filename_ok": True
            })

    scan(ORG_DIR,  "genuine", PAT_ORG)
    scan(FORG_DIR, "forgery", PAT_FORG)

    df = pd.DataFrame(rows)
    return df

df = build_metadata()

print("Total files:", len(df))
print("Bad filenames:", (~df["filename_ok"]).sum())
print(df["label"].value_counts(dropna=False))


# =========================
# 1) Helpers to read images
# =========================
def read_gray(path: str) -> np.ndarray:
    """Read image as grayscale uint8."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img

def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """Return binary image (foreground=1, background=0) using Otsu.
    Assumes background is light and ink is dark (common for signatures)."""
    # Otsu gives thresholded image; we invert so ink becomes 1
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (bw == 0).astype(np.uint8)  # dark pixels -> ink
    return ink


# =================================
# 2) Image-level EDA (fast sampling)
# =================================
# Reading *all* images is fine for CEDAR (~2640), but we keep it efficient.
valid = df[df["filename_ok"]].copy()

# Compute basic stats
records = []
for p in valid["path"].values:
    g = read_gray(p)
    h, w = g.shape
    records.append({
        "path": p,
        "height": h,
        "width": w,
        "min": int(g.min()),
        "max": int(g.max()),
        "mean": float(g.mean()),
        "std": float(g.std())
    })

img_stats = pd.DataFrame(records)
valid = valid.merge(img_stats, on="path", how="left")

print("\nImage size (height,width) unique counts (top 10):")
print(valid.groupby(["height", "width"]).size().sort_values(ascending=False).head(10))


# =========================
# 3) Ink density (Otsu)
# =========================
# Ink density = fraction of pixels classified as ink (foreground)
ink_density = []
for p in valid["path"].values:
    g = read_gray(p)
    ink = otsu_binarize(g)
    ink_density.append(float(ink.mean()))  # since ink is 0/1

valid["ink_density"] = ink_density

print("\nInk density summary:")
print(valid["ink_density"].describe())


# =========================
# 4) Plots (matplotlib only)
# =========================
OUT_DIR = Path("cedar_eda_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=200, bbox_inches="tight")
    plt.close()

# 4.1 Height distribution
plt.figure()
plt.hist(valid["height"].dropna(), bins=30)
plt.title("CEDAR: Image Height Distribution")
plt.xlabel("Height (pixels)")
plt.ylabel("Count")
savefig("01_height_distribution.png")

# 4.2 Width distribution
plt.figure()
plt.hist(valid["width"].dropna(), bins=30)
plt.title("CEDAR: Image Width Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Count")
savefig("02_width_distribution.png")

# 4.3 Mean intensity distribution (grayscale)
plt.figure()
plt.hist(valid["mean"].dropna(), bins=30)
plt.title("CEDAR: Mean Pixel Intensity (Grayscale)")
plt.xlabel("Mean intensity (0=dark, 255=bright)")
plt.ylabel("Count")
savefig("03_mean_intensity_distribution.png")

# 4.4 Ink density distribution
plt.figure()
plt.hist(valid["ink_density"].dropna(), bins=30)
plt.title("CEDAR: Ink Density Distribution (Otsu)")
plt.xlabel("Ink density (fraction of ink pixels)")
plt.ylabel("Count")
savefig("04_ink_density_distribution.png")

# 4.5 Compare genuine vs forgery ink density (boxplot)
plt.figure()
data = [
    valid.loc[valid["label"] == "genuine", "ink_density"].dropna().values,
    valid.loc[valid["label"] == "forgery", "ink_density"].dropna().values
]
plt.boxplot(data, labels=["genuine", "forgery"])
plt.title("CEDAR: Ink Density by Class (Otsu)")
plt.ylabel("Ink density")
savefig("05_ink_density_boxplot_by_label.png")

print(f"\nSaved plots to: {OUT_DIR.resolve()}")


# ==========================================
# 5) Visual EDA: sample grids (very useful)
# ==========================================
def show_grid(image_paths, nrows, ncols, title, save_name, preprocess=None):
    plt.figure(figsize=(2.5*ncols, 2.5*nrows))
    for i, p in enumerate(image_paths[: nrows*ncols]):
        img = read_gray(p)
        if preprocess is not None:
            img = preprocess(img)
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(Path(p).name, fontsize=9)
    plt.suptitle(title, fontsize=14)
    savefig(save_name)

# 5.1 Random samples: genuine + forgery
rng = np.random.default_rng(42)
sample_g = valid[valid["label"] == "genuine"].sample(8, random_state=42)["path"].tolist()
sample_f = valid[valid["label"] == "forgery"].sample(8, random_state=42)["path"].tolist()

show_grid(sample_g, 2, 4, "Random Genuine Samples (CEDAR)", "06_random_genuine_grid.png")
show_grid(sample_f, 2, 4, "Random Forgery Samples (CEDAR)", "07_random_forgery_grid.png")

# 5.2 Intra-class variability: 8 genuine signatures from the same writer
writer_pick = int(valid["writer_id"].dropna().unique()[0])  # you can change this
same_writer_g = (
    valid[(valid["writer_id"] == writer_pick) & (valid["label"] == "genuine")]
    .sort_values("sample_id")["path"].tolist()
)
show_grid(
    same_writer_g, 2, 4,
    f"Intra-class Variability: Writer {writer_pick} (Genuine)",
    "08_intra_class_variability_writer_genuine.png"
)

# 5.3 Inter-class similarity: Genuine vs Forgery for same writer (paired by sample_id if possible)
gw = valid[(valid["writer_id"] == writer_pick) & (valid["label"] == "genuine")][["sample_id","path"]]
fw = valid[(valid["writer_id"] == writer_pick) & (valid["label"] == "forgery")][["sample_id","path"]]
paired = gw.merge(fw, on="sample_id", suffixes=("_genuine", "_forgery")).sort_values("sample_id")

# Build an alternating list: genuine, forgery, genuine, forgery...
pairs_list = []
for _, r in paired.head(4).iterrows():  # show 4 pairs (8 images)
    pairs_list.append(r["path_genuine"])
    pairs_list.append(r["path_forgery"])

show_grid(
    pairs_list, 2, 4,
    f"Inter-class Similarity: Writer {writer_pick} (Genuine vs Skilled Forgery)",
    "09_inter_class_similarity_writer_pairs.png"
)

# 5.4 Show effect of Otsu binarization (before vs after) on a few samples
demo_paths = valid.sample(4, random_state=7)["path"].tolist()

# Before
show_grid(demo_paths, 1, 4, "Preprocessing Demo: Original Grayscale", "10_preprocess_demo_grayscale.png")

# After (ink mask)
def ink_to_display(gray):
    ink = otsu_binarize(gray) * 255
    return 255 - ink  # show ink as dark for readability

show_grid(demo_paths, 1, 4, "Preprocessing Demo: Otsu Binarized", "11_preprocess_demo_otsu.png")

print("Saved sample grids and preprocessing demos.")


# =========================
# 6) Simple EDA tables
# =========================
# Per-writer counts (should be 24 & 24)
per_writer = (
    valid.groupby(["writer_id", "label"])["path"].count()
    .unstack(fill_value=0)
    .reset_index()
)
per_writer["total"] = per_writer.get("genuine", 0) + per_writer.get("forgery", 0)

per_writer.to_csv(OUT_DIR / "per_writer_counts.csv", index=False)
valid.to_csv(OUT_DIR / "metadata_with_stats.csv", index=False)

issues = per_writer[(per_writer["genuine"] != 24) | (per_writer["forgery"] != 24)]
issues.to_csv(OUT_DIR / "writer_count_issues.csv", index=False)

print(f"\nSaved CSVs:")
print(f"- {OUT_DIR / 'metadata_with_stats.csv'}")
print(f"- {OUT_DIR / 'per_writer_counts.csv'}")
print(f"- {OUT_DIR / 'writer_count_issues.csv'}")
