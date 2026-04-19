import os
import cv2
import numpy as np
from collections import defaultdict

# Paths
ann_dir = "/home/adamm/Documents/FSOD/Data/Lavyanut/new_gt/"
img_dir = "/home/adamm/Documents/FSOD/Data/Lavyanut/images/"
out_img_path = "/home/adamm/Documents/FSOD/Presentations/best_sample.png"
out_txt_path = "/home/adamm/Documents/FSOD/Presentations/best_sample_stats.txt"

CLASS_MAPPING = {
    '0': 'ExtremelyLongHeavyDuty',
    '1': 'LongHeavyDuty',
    '2': 'HeavyDuty',
    '3': 'MediumStandard',
    '4': 'MediumSmall',
    '5': 'Small',
    '6': 'ExtremelyLongHeavyDutyTraileronly',
    '7': 'HeavyDutyTractorTruck',
    '8': 'CementMixerTrucks',
    '9': 'Bulldozers',
    '10': 'MobileCranes',
    '11': 'Forklifts',
    '12': 'TruckTractor',
    '13': 'Other'
}

# ---------- Step 1: Find best file ----------
best_file = None
best_unique_classes = 0
best_lines = []

for file in os.listdir(ann_dir):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(ann_dir, file)
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    classes = [line.split()[0] for line in lines]
    unique_classes = set(classes)

    if len(unique_classes) > best_unique_classes:
        best_unique_classes = len(unique_classes)
        best_file = file
        best_lines = lines

print(f"Selected file: {best_file} ({best_unique_classes} classes)")

# ---------- Step 2: Load image robustly ----------
base_name = os.path.splitext(best_file)[0]
img_path = None

for ext in [".jpg", ".JPG", ".png", ".jpeg"]:
    candidate = os.path.join(img_dir, base_name + ext)
    if os.path.exists(candidate):
        img_path = candidate
        break

if img_path is None:
    raise ValueError(f"No matching image for {best_file}")

img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Failed to load image: {img_path}")

h, w = img.shape[:2]

# ---------- Step 3: Draw + collect stats ----------
class_counts = defaultdict(int)
class_areas = defaultdict(list)

for line in best_lines:
    parts = line.split()
    cls = parts[0]
    coords = list(map(float, parts[1:]))

    # convert normalized → pixels
    pts = []
    for i in range(0, len(coords), 2):
        x = coords[i] * w
        y = coords[i+1] * h
        pts.append([x, y])

    pts = np.array(pts, dtype=np.float32)

    # --- area (polygon area using Shoelace formula via OpenCV) ---
    area = cv2.contourArea(pts)

    class_counts[cls] += 1
    class_areas[cls].append(area)

    # --- draw ---
    pts_int = pts.astype(np.int32)
    color = tuple(int(c) for c in np.random.RandomState(int(cls)).randint(0, 255, 3))

    cv2.polylines(img, [pts_int], True, color, 20)

    label = CLASS_MAPPING.get(cls, cls)
    cv2.putText(img, label, tuple(pts_int[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# ---------- Step 4: Save image ----------
cv2.imwrite(out_img_path, img)

# ---------- Step 5: Save stats ----------
with open(out_txt_path, "w") as f:
    f.write(f"Source annotation file: {best_file}\n")
    f.write(f"Image path: {img_path}\n\n")

    f.write("Class statistics:\n")
    f.write("--------------------------------------\n")

    for cls in sorted(class_counts.keys(), key=lambda x: int(x)):
        count = class_counts[cls]
        avg_area = np.mean(class_areas[cls])

        name = CLASS_MAPPING.get(cls, cls)

        f.write(f"Class {cls} ({name}):\n")
        f.write(f"  Count: {count}\n")
        f.write(f"  Avg area (pixels): {avg_area:.2f}\n\n")

print(f"Saved image to: {out_img_path}")
print(f"Saved stats to: {out_txt_path}")