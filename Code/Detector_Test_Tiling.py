import os
import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

def process_image_with_tiling(image_path, model, output_dir, tile_size=1024, overlap=0.20):
    """
    Splits a large image into tiles, runs YOLO inference, stitches predictions,
    applies safe NMS to prevent deleting dense vehicles, and saves visualization.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}. Skipping...")
        return

    img_h, img_w = img.shape[:2]
    
    # Calculate stride based on overlap
    stride = int(tile_size * (1 - overlap))
    
    # Dynamically calculate thickness so it's visible on massive 8k images
    thickness = max(2, int(max(img_h, img_w) / 1500))
    
    print(f"\nProcessing Image: {os.path.basename(image_path)}")
    print(f"Original size: {img_w}x{img_h}. Tiling with {tile_size}x{tile_size} and {overlap*100:.0f}% overlap...")

    all_boxes =[]     # For NMS calculation (Standard Axis-Aligned Boxes)
    all_scores =[]    # Confidence scores
    all_classes = []   # Class IDs
    all_raw_preds =[] # To store the original prediction shapes (OBB points or standard box)
    is_obb = False     # Flag to track if the model outputs OBB

    # Generate starting coordinates for tiles
    y_starts = list(range(0, img_h, stride))
    x_starts = list(range(0, img_w, stride))

    # 1. Tile the image and predict
    for y in y_starts:
        for x in x_starts:
            # Determine tile boundaries
            y1, y2 = y, min(y + tile_size, img_h)
            x1, x2 = x, min(x + tile_size, img_w)

            # Fix edge cases: If the remaining edge is less than 1024, step backward 
            # to guarantee the crop is EXACTLY 1024x1024 to match your training size
            if y2 - y1 < tile_size and img_h >= tile_size:
                y1 = max(0, img_h - tile_size)
                y2 = y1 + tile_size
            if x2 - x1 < tile_size and img_w >= tile_size:
                x1 = max(0, img_w - tile_size)
                x2 = x1 + tile_size

            # Crop the tile
            tile = img[y1:y2, x1:x2]
            
            if tile.size == 0:
                continue

            # PREDICT: Added conf=0.15 to prevent small cars from being skipped
            results = model.predict(tile, imgsz=tile_size, conf=0.0001, verbose=False)
            result = results[0]

            # 2. Extract predictions and shift them to the global coordinate system
            if hasattr(result, 'obb') and result.obb is not None:
                is_obb = True
                if len(result.obb) > 0:
                    obbs = result.obb.xyxyxyxy.cpu().numpy()
                    confs = result.obb.conf.cpu().numpy()
                    cls_ids = result.obb.cls.cpu().numpy()

                    for obb, conf, cls_id in zip(obbs, confs, cls_ids):
                        # Shift the 4 polygon corners by the tile's top-left coordinates
                        global_obb = obb + np.array([x1, y1])
                        
                        # Compute axis-aligned bounding box solely for duplicate NMS
                        min_x, min_y = np.min(global_obb, axis=0)
                        max_x, max_y = np.max(global_obb, axis=0)
                        
                        # Prevent 0-area boxes
                        max_x = max(min_x + 1, max_x)
                        max_y = max(min_y + 1, max_y)
                        
                        all_boxes.append([min_x, min_y, max_x, max_y])
                        all_scores.append(conf)
                        all_classes.append(cls_id)
                        all_raw_preds.append(global_obb)
                        
            elif hasattr(result, 'boxes') and result.boxes is not None:
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy()

                    for box, conf, cls_id in zip(boxes, confs, cls_ids):
                        global_box = box + np.array([x1, y1, x1, y1])
                        x_min, y_min, x_max, y_max = global_box
                        
                        x_max = max(x_min + 1, x_max)
                        y_max = max(y_min + 1, y_max)
                        
                        all_boxes.append([x_min, y_min, x_max, y_max])
                        all_scores.append(conf)
                        all_classes.append(cls_id)
                        all_raw_preds.append(global_box)

    print(f"Total raw detections across all tiles: {len(all_boxes)}")

    # 3. Global Non-Maximum Suppression (NMS)
    if len(all_boxes) > 0:
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        classes_tensor = torch.tensor(all_classes, dtype=torch.float32)

        # INCREASED THRESHOLD: 0.65 ensures that adjacent densely parked cars are 
        # NOT deleted, but identical duplicates split across tiles still merge perfectly.
        keep_indices = torchvision.ops.batched_nms(boxes_tensor, scores_tensor, classes_tensor, iou_threshold=0.65)
        
        print(f"Detections remaining after global NMS: {len(keep_indices)}")

        # 4. Visualize the finalized predictions
        for idx in keep_indices:
            idx = int(idx)
            raw_pred = all_raw_preds[idx]
            cls_id = int(all_classes[idx])
            
            # Green or Blue alternating based on Class ID
            color = (0, 255, 0) if cls_id % 2 == 0 else (255, 0, 0) 
            
            if is_obb:
                # Draw oriented polygon
                pts = np.int32(raw_pred).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            else:
                # Draw standard rectangle
                x_min, y_min, x_max, y_max = map(int, raw_pred)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    else:
        print("No detections found! Try lowering the 'conf' threshold in model.predict().")

    # 5. Save the final annotated image
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"stitched_{img_name}")
    cv2.imwrite(save_path, img)
    print(f"Saved annotated image to: {save_path}")

def main():
    weights_path = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Outputs/YOLO26_Train_FAIR1M/fair1m_vehicle_training/weights/best.pt'
    output_dir = '/home/adamm/Documents/FSOD/Data/Tagged_Data/custom_tiled_predictions/'
    input_dir = '/home/adamm/Documents/FSOD/Data/Tagged_Data/images/'

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return

    print(f"Loading YOLO Model from {weights_path}...")
    model = YOLO(weights_path)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    image_files =[f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No valid image files found in {input_dir}.")
        return

    print(f"Found {len(image_files)} images. Starting batch processing...")

    for img_file in image_files:
        full_image_path = os.path.join(input_dir, img_file)
        
        process_image_with_tiling(
            image_path=full_image_path,
            model=model,
            output_dir=output_dir,
            tile_size=1024,
            overlap=0.20
        )
        
    print("\n=============================================")
    print(f"Batch processing complete! All results are in:\n{output_dir}")
    print("=============================================")

if __name__ == '__main__':
    main()