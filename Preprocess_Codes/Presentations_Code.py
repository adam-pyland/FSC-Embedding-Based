import os
import cv2
import numpy as np
from collections import defaultdict

FIND_BY_TARGET_CLASS = False  # False = original behavior, True = maximize class '6'
TARGET_CLASS = '6'  # ExtremelyLongHeavyDutyTraileronly

WORK_PLACE = 'yehud' # The place where I am working in: 'yehud' or 'matrix'

data_path = r'C:\Adams\FSOD' if WORK_PLACE is 'yehud' else '/home/adamm/Documents/FSOD'


# Paths
ann_dir = f"{data_path}/Data/Lavyanut/Lavyanut/new_gt/"
img_dir = f"{data_path}/Data/Lavyanut/Lavyanut/images/"
out_img_path = f"{data_path}/Presentations/best_sample.png"
out_txt_path = f"{data_path}/Presentations/best_sample_stats.txt"

out_dir = os.path.dirname(out_img_path)
crops_dir = os.path.join(out_dir, "crops")
os.makedirs(crops_dir, exist_ok=True)




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
best_score = -1
best_lines = []

class_colors = {}
present_classes = set()

for file in os.listdir(ann_dir):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(ann_dir, file)
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    classes = [line.split()[0] for line in lines]

    if FIND_BY_TARGET_CLASS:
        # Count how many instances of TARGET_CLASS
        score = sum(1 for c in classes if c == TARGET_CLASS)
    else:
        # Original behavior: maximize number of unique classes
        score = len(set(classes))

    if score > best_score:
        best_score = score
        best_file = file
        best_lines = lines

if FIND_BY_TARGET_CLASS:
    print(f"Selected file: {best_file} ({best_score} instances of class {TARGET_CLASS})")
else:
    print(f"Selected file: {best_file} ({best_score} unique classes)")

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

TARGET_CLASS = '6'  # ExtremelyLongHeavyDutyTraileronly
crop_idx = 0
img_original = img.copy()
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

    # --- area ---
    area = cv2.contourArea(pts)

    class_counts[cls] += 1
    class_areas[cls].append(area)

    # --- drawing thickness logic ---
    if cls == TARGET_CLASS:
        thickness = 20   # thicker for target class
    else:
        thickness = 10

    pts_int = pts.astype(np.int32)
    color = tuple(int(c) for c in np.random.RandomState(int(cls)).randint(0, 255, 3))

    class_colors[cls] = color
    present_classes.add(cls)

    cv2.polylines(img, [pts_int], True, color, thickness)

    label = CLASS_MAPPING.get(cls, cls)
    cv2.putText(img, label, tuple(pts_int[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # --- crop logic for target class ---
    if cls == TARGET_CLASS:
        # --- get rotated rectangle ---
        rect = cv2.minAreaRect(pts_int)
        (cx, cy), (rw, rh), angle = rect

        # --- fix angle to make object horizontal ---
        if rw < rh:
            angle += 90
            rw, rh = rh, rw

        # --- rotate entire image ---
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img_original, M, (w, h))

        # --- crop aligned rectangle ---
        x1 = int(cx - rw / 2)
        y1 = int(cy - rh / 2)
        x2 = int(cx + rw / 2)
        y2 = int(cy + rh / 2)

        # clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = rotated[y1:y2, x1:x2]

        # --- OPTIONAL: enforce horizontal output ---
        if crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        crop_filename = f"{base_name}_crop_{crop_idx}.png"
        crop_path = os.path.join(crops_dir, crop_filename)

        cv2.imwrite(crop_path, crop)
        crop_idx += 1

        crop_filename = f"{base_name}_crop_{crop_idx}.png"
        crop_path = os.path.join(crops_dir, crop_filename)

        cv2.imwrite(crop_path, crop)
        crop_idx += 1

# ---------- Step 3.5: Draw legend ----------
legend_items = sorted(present_classes, key=lambda x: int(x))

# layout parameters
scale = 4.0  # 👈 increase this (e.g., 1.5, 2, 3)

padding = int(10 * scale)
box_size = int(20 * scale)
line_height = int(30 * scale)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5 * scale
thickness = max(1, int(1 * scale))

text_sizes = []
for cls in legend_items:
    label = CLASS_MAPPING.get(cls, cls)
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_sizes.append((tw, th))

max_text_width = max(tw for tw, _ in text_sizes) if text_sizes else 0

legend_width = padding * 3 + box_size + max_text_width
legend_height = padding * 2 + line_height * len(legend_items)

# top-right corner placement
x_start = w - legend_width - 10
y_start = 10

# optional: semi-transparent background
overlay = img.copy()
cv2.rectangle(
    overlay,
    (x_start, y_start),
    (x_start + legend_width, y_start + legend_height),
    (0, 0, 0),
    -1
)
alpha = 0.4
cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# draw each class entry
for i, cls in enumerate(legend_items):
    color = class_colors[cls]
    label = CLASS_MAPPING.get(cls, cls)

    y = y_start + padding + i * line_height + box_size

    # color box
    cv2.rectangle(
        img,
        (x_start + padding, y - box_size),
        (x_start + padding + box_size, y),
        color,
        -1
    )

    # text
    cv2.putText(
        img,
        label,
        (x_start + padding * 2 + box_size, y - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

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