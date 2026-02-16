import os
import cv2

# =======================
# PATHS (EDIT IF NEEDED)
# =======================
IMG_DIR = '/work/Sultan/data/pickup/images/'
SRC_LABEL_DIR = '/work/Sultan/data/pickup/labels/'
DST_LABEL_DIR = '/work/Sultan/data/pickup/labels/yolo_val'

# =======================
# CONVERSION FUNCTION
# =======================
def convert_visdrone_to_yolo(img_dir, src_label_dir, dst_label_dir):
    os.makedirs(dst_label_dir, exist_ok=True)

    img_exts = ('.jpg', '.jpeg', '.png')

    for label_file in os.listdir(src_label_dir):
        if not label_file.endswith('.txt'):
            continue

        base_name = os.path.splitext(label_file)[0]

        # Find corresponding image (for size only)
        img_path = None
        for ext in img_exts:
            candidate = os.path.join(img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"[WARN] Image not found for {label_file}, skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read {img_path}, skipping.")
            continue

        img_h, img_w = img.shape[:2]
        yolo_lines = []

        with open(os.path.join(src_label_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue

                # VisDrone fields
                left   = float(parts[0])
                top    = float(parts[1])
                width  = float(parts[2])
                height = float(parts[3])
                category = int(parts[5])

                # Keep categories 1–10 only
                if category < 1 or category > 10:
                    continue

                yolo_class = category - 1

                x_center = (left + width / 2) / img_w
                y_center = (top + height / 2) / img_h
                w_norm = width / img_w
                h_norm = height / img_h

                yolo_lines.append(
                    f"{yolo_class} {x_center:.6f} {y_center:.6f} "
                    f"{w_norm:.6f} {h_norm:.6f}"
                )

        # Write YOLO label file
        dst_path = os.path.join(dst_label_dir, label_file)
        with open(dst_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    print("✅ YOLO label conversion finished.")


# =======================
# SCRIPT ENTRY POINT
# =======================
if __name__ == "__main__":
    convert_visdrone_to_yolo(
        IMG_DIR,
        SRC_LABEL_DIR,
        DST_LABEL_DIR
    )
