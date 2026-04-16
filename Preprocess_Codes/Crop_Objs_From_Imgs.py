import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image


def copy_imgs_by_txt_list():
    # Paths
    txt_file = "/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/Image_lists/Val_with_BCV.txt"
    src_dir = "/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/Images/Val/"
    dst_dir = "/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/Vehicles_Novel_Class"

    os.makedirs(dst_dir, exist_ok=True)

    # Read base names (without extension)
    with open(txt_file, "r") as f:
        target_names = set(line.strip() for line in f if line.strip())

    # Index source files by base name
    src_files = {}
    for filename in os.listdir(src_dir):
        base, ext = os.path.splitext(filename)
        src_files[base] = filename  # keeps full filename with extension

    # Copy matches
    for name in target_names:
        if name in src_files:
            src_path = os.path.join(src_dir, src_files[name])
            dst_path = os.path.join(dst_dir, src_files[name])
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Not found: {name}")

    print("Done.")



def crop_objects_from_xml():
    # --- Configuration ---
    xml_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/VOC_Annotations/Horizontal_Bounding_Boxes/Val/'
    img_dir = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/Vehicles_Novel_Class'
    out_Novel_dir = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/Original_Novel_Class_Img_Crops'
    out_Base_dir = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/Original_Base_Class_Img_Crops'

    base_classes = [
        'Small Car', 'Bus', 'Dump Truck', 'Van', 'Tractor', 
        'Truck Tractor', 'Excavator', 'other-vehicle'
    ]
    novel_classes = [
        'Cargo Truck', 'Trailer'
    ]

    # Create output directory if it doesn't exist
    if not os.path.exists(out_Novel_dir):
        os.makedirs(out_Novel_dir)
        print(f"Created output directory: {out_Novel_dir}")

    if not os.path.exists(out_Base_dir):
        os.makedirs(out_Base_dir)
        print(f"Created output directory: {out_Base_dir}")

    # Get list of images
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
    
    print(f"Found {len(image_files)} images. Starting processing...")

    for img_file in image_files:
        # Get the filename without extension (e.g., 'v_0' from 'v_0.jpg')
        file_id = os.path.splitext(img_file)[0]
        
        # Construct path to corresponding XML
        xml_path = os.path.join(xml_dir, file_id + '.xml')
        img_path = os.path.join(img_dir, img_file)

        # Check if XML annotation exists
        if not os.path.exists(xml_path):
            # print(f"No XML found for {img_file}, skipping.")
            continue

        try:
            # Load the image
            image = Image.open(img_path)
            
            # Parse the XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Dictionary to keep track of object counts per class for THIS image
            # e.g., {'Small Car': 1, 'Bus': 1, ...}
            class_base_counters = {cls: 1 for cls in base_classes}
            class_novel_counters = {cls: 1 for cls in novel_classes}

            for obj in root.findall('object'):
                class_name = obj.find('name').text

                if class_name in base_classes or class_name in novel_classes:
                    # Get Bounding Box Coordinates
                    bndbox = obj.find('bndbox')
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))

                    # Crop the image (box definition: left, upper, right, lower)
                    cropped_img = image.crop((xmin, ymin, xmax, ymax))

                    # create safe filename (replace spaces in class name with underscores if desired, 
                    # though here I keep it as string formatted or replace for safety)
                    safe_class_name = class_name.replace(" ", "_")
                    
                    # Naming convention: <img name>_<object class name>_<object number>.jpg
                    if class_name in base_classes:
                        current_count = class_base_counters[class_name]
                        out_dir = out_Base_dir
                        # Increment the counter for this specific class
                        class_base_counters[class_name] += 1
                    else:
                        current_count = class_novel_counters[class_name]
                        out_dir = out_Novel_dir
                        class_novel_counters[class_name] += 1
                    
                    new_filename = f"{file_id}_{safe_class_name}_{current_count}.jpg"                    
                    save_path = os.path.join(out_dir, new_filename)

                    # Convert to RGB if necessary (e.g. if original was RGBA or P) to save as JPG
                    if cropped_img.mode in ("RGBA", "P"):
                        cropped_img = cropped_img.convert("RGB")

                    cropped_img.save(save_path)
               

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    print("Processing complete.")


import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from tqdm import tqdm

def crop_objects_from_FAIR1M_xml_obb():
    xml_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/VOC_Annotations/Horizontal_Bounding_Boxes/Train/'
    img_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/images/train/'
    out_Novel_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/Image_Obj_Crops/train/'
    out_Base_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/Image_Obj_Crops/train/'

    base_classes = ['Small Car', 'Bus', 'Dump Truck', 'Van', 'Tractor', 'Truck Tractor', 'Excavator', 'other-vehicle']
    novel_classes = ['Cargo Truck', 'Trailer']

    os.makedirs(out_Novel_dir, exist_ok=True)
    os.makedirs(out_Base_dir, exist_ok=True)

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
    print(f"Found {len(image_files)} images. Starting OBB processing...")

    class_counters = {cls: 1 for cls in base_classes + novel_classes}

    for img_file in tqdm(image_files):
        file_id = os.path.splitext(img_file)[0]
        xml_path = os.path.join(xml_dir, file_id + '.xml')
        img_path = os.path.join(img_dir, img_file)

        if not os.path.exists(xml_path):
            continue

        try:
            # Using cv2 to read the image for perspective warping
            image = cv2.imread(img_path)
            if image is None:
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text

                if class_name in base_classes or class_name in novel_classes:
                    polygon = obj.find('polygon')
                    if polygon is None:
                        continue # Skip if no polygon exists

                    # Extract the 4 polygon points
                    pts = []
                    for i in range(4):
                        x = float(polygon.find(f'x{i}').text)
                        y = float(polygon.find(f'y{i}').text)
                        pts.append([x, y])
                    
                    src_pts = np.array(pts, dtype=np.float32)

                    # Calculate width and height of the bounding box
                    # distance between pt0 and pt1 is one edge, pt1 and pt2 is the other
                    width = int(np.linalg.norm(src_pts[0] - src_pts[1]))
                    height = int(np.linalg.norm(src_pts[1] - src_pts[2]))

                    # Create destination points for a straight rectangle
                    dst_pts = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype=np.float32)

                    # Get the perspective transform matrix and warp the image
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    cropped_img = cv2.warpPerspective(image, M, (width, height))

                    safe_class_name = class_name.replace(" ", "_")
                    out_dir = out_Base_dir if class_name in base_classes else out_Novel_dir
                    
                    current_count = class_counters[class_name]
                    class_counters[class_name] += 1
                    
                    new_filename = f"{file_id}_{safe_class_name}_{current_count}.jpg"                    
                    save_path = os.path.join(out_dir, new_filename)

                    # Save the cropped image
                    cv2.imwrite(save_path, cropped_img)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    print("OBB Processing complete.")


def crop_objects_from_Lavyanut_txt_obb():
    # --- CHANGED: Renamed directory variables to match TXT processing ---
    txt_dir = '/home/adamm/Documents/FSOD/Data/Lavyanut/new_gt/'
    img_dir = '/home/adamm/Documents/FSOD/Data/Lavyanut/images/'
    
    # --- CHANGED: Unified output directory since new base/novel splits weren't provided ---
    out_dir = '/home/adamm/Documents/FSOD/Data/Lavyanut/Obj_Crops/'
    os.makedirs(out_dir, exist_ok=True)

    # --- CHANGED: Added your exact CLASS_MAPPING dictionary ---
    CLASS_MAPPING = {
        '0': 'Extremely Long Heavy-Duty',
        '1': 'Long Heavy- Duty',
        '2': 'Heavy- Duty',
        '3': 'Medium- Standard',
        '4': 'Medium- Small',
        '5': 'Small',
        '6': 'Extremely Long Heavy-Duty, Trailer only',
        '7': 'Heavy Duty Tractor Truck',
        '8': 'Cement Mixer Trucks',
        '9': 'Bulldozers',
        '10': 'Mobile Cranes',
        '11': 'Forklifts',
        '12': 'Truck Tractor',
        '13': 'Other'
    }

    image_files =[f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
    print(f"Found {len(image_files)} images. Starting OBB processing...")

    # --- CHANGED: Initialize counters based on the new dictionary values ---
    class_counters = {cls: 1 for cls in CLASS_MAPPING.values()}

    for img_file in tqdm(image_files):
        file_id = os.path.splitext(img_file)[0]
        
        # --- CHANGED: Look for .txt extension instead of .xml ---
        txt_path = os.path.join(txt_dir, file_id + '.txt')
        img_path = os.path.join(img_dir, img_file)

        if not os.path.exists(txt_path):
            continue

        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            # --- CHANGED: Grab image dimensions to un-normalize the TXT coordinates ---
            img_height, img_width = image.shape[:2]

            # --- CHANGED: Open and read lines from the TXT file instead of parsing XML ---
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue # Skip malformed lines

                class_id = parts[0]
                if class_id not in CLASS_MAPPING:
                    continue # Skip if class isn't in our dictionary

                class_name = CLASS_MAPPING[class_id]

                # --- CHANGED: Extract points and multiply by image width/height ---
                pts =[]
                for i in range(4):
                    # TXT format: class_id x0 y0 x1 y1 x2 y2 x3 y3
                    x = float(parts[1 + (i * 2)]) * img_width
                    y = float(parts[2 + (i * 2)]) * img_height
                    pts.append([x, y])
                
                src_pts = np.array(pts, dtype=np.float32)

                # Calculate width and height of the bounding box
                width = int(np.linalg.norm(src_pts[0] - src_pts[1]))
                height = int(np.linalg.norm(src_pts[1] - src_pts[2]))
                
                # Failsafe for 0-dimension crops
                if width <= 0 or height <= 0:
                    continue

                # Create destination points for a straight rectangle
                dst_pts = np.array([[0, 0],
                    [width - 1, 0],[width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)

                # Get the perspective transform matrix and warp the image
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                cropped_img = cv2.warpPerspective(image, M, (width, height))

                # --- CHANGED: Replaced replacing spaces with underscores to replacing with nothing (no spacing) ---
                safe_class_name = class_name.replace(" ", "")
                
                current_count = class_counters[class_name]
                class_counters[class_name] += 1
                
                # --- CHANGED: Saving directly to out_dir ---
                new_filename = f"{file_id}_{safe_class_name}_{current_count}.jpg"                    
                save_path = os.path.join(out_dir, new_filename)

                # Save the cropped image
                cv2.imwrite(save_path, cropped_img)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    print("OBB Processing complete.")



if __name__ == "__main__":
    # copy_imgs_by_txt_list()
    # crop_objects_from_xml()
    # crop_objects_from_FAIR1M_xml_obb()
    crop_objects_from_Lavyanut_txt_obb()