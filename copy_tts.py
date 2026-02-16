import os
import cv2

def convert_visdrone_to_yolo(img_folder, label_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Supported image extensions
    img_exts = ('.jpg', '.jpeg', '.png')
    
    # Process each label file
    for label_file in os.listdir(label_folder):
        if not label_file.endswith('.txt'):
            continue
            
        # Match label to image to get dimensions
        base_name = os.path.splitext(label_file)[0]
        img_path = None
        for ext in img_exts:
            temp_path = os.path.join(img_folder, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if img_path is None:
            print(f"Warning: Image for {label_file} not found. Skipping.")
            continue

        # Get image size for normalization
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        yolo_lines = []
        with open(os.path.join(label_folder, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                
                # VisDrone format: left, top, width, height, score, category, trunc, occl
                left = float(parts[0])
                top = float(parts[1])
                width = float(parts[2])
                height = float(parts[3])
                category = int(parts[5])
                
                # Skip 'ignored regions' (0) and 'others' (11) if desired
                # Here we convert category 1-10 to 0-9
                if category < 1 or category > 10:
                    continue
                yolo_class = category - 1 

                # Calculate YOLO normalized coordinates
                # Center X, Center Y, Width, Height
                x_center = (left + width / 2) / img_w
                y_center = (top + height / 2) / img_h
                w_norm = width / img_w
                h_norm = height / img_h

                yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save to new destination
        with open(os.path.join(output_folder, label_file), 'w') as f:
            f.write('\n'.join(yolo_lines))

# Usage
IMG_DIR = '/work/Sultan/data/nopickup/images/'
LABEL_DIR = '/work/Sultan/data/nopickup/labels/'
DEST_DIR = '/work/Sultan/data/nopickup/labels/train_yolo'

convert_visdrone_to_yolo(IMG_DIR, LABEL_DIR, DEST_DIR)
print("Conversion complete!")