import os
import numpy as np  # Added for custom visualization
from PIL import Image  # Added to check image resolution
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction  # Added get_prediction for non-tiled inference
from sahi.utils.cv import visualize_object_predictions  # Added to support custom bounding box colors

def main():
    model_path = "/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Outputs/YOLO26_Train_FAIR1M/fair1m_vehicle_training/weights/best.pt"
    images_dir = "/home/adamm/Documents/FSOD/Data/Tagged_Data/images/"
    output_dir = "/home/adamm/Documents/FSOD/Data/Tagged_Data/sahi_tiled_predictions"
    
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model into SAHI...")
    # SAHI uses "yolov8" as the base wrapper for all recent Ultralytics models (including custom ones)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", 
        model_path=model_path,
        confidence_threshold=0.25,
        device="cuda:0" # Change to "cpu" if running on a machine without a GPU
    )

    print(f"Running SAHI sliced inference on images in: {images_dir}")
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(images_dir, img_name)
        print(f"Slicing and processing: {img_name}")

        # Check resolution
        with Image.open(img_path) as img:
            width, height = img.size

        # 1. Run Inference (Tiling vs Standard)
        # If either width or height is strictly greater than 1024, use slicing
        if max(width, height) > 1024:
            # This chops the image into 640x640 tiles, predicts, and merges them using NMS
            result = get_sliced_prediction(
                img_path,
                detection_model,
                slice_height=1024,          # Multiple of 32
                slice_width=1024,           # Multiple of 32
                overlap_height_ratio=0.2,  # 20% overlap ensures vehicles on the edge of a tile aren't cut in half
                overlap_width_ratio=0.2
            )
        else:
            # If resolution is 1024 or lower, don't do the tiling
            result = get_prediction(
                img_path,
                detection_model
            )

        # 2. Export visualizations
        # We use visualize_object_predictions directly instead of result.export_visuals() 
        # because the default export_visuals() doesn't expose a 'color' parameter.
        visualize_object_predictions(
            image=np.ascontiguousarray(result.image),
            object_prediction_list=result.object_prediction_list,
            rect_th=2,                 # Changed from 1 to 2 for a slightly bigger thickness
            text_size=0.3,             # Smaller text labels (unchanged)
            text_th=None,
            color=(0, 255, 0),         # Set color to Green (RGB)
            hide_labels=False,         
            hide_conf=True,            # Hides the confidence numbers (e.g., "0.84")
            output_dir=output_dir,
            file_name=img_name.split('.')[0],
            export_format="png"
        )

    print(f"\n✅ SAHI Tiled Testing Complete!")
    print(f"Visualizations with thin boxes saved to: {output_dir}")

if __name__ == '__main__':
    main()