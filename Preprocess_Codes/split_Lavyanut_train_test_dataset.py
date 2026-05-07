import os
import glob
import re
import shutil
import random
from collections import Counter

# ================= Paths & Settings =================
SOURCE_DIR = "/home/adamm/Documents/FSOD/Data/Lavyanut_partial/Obj_Embs/All_Embs/"
BASE_OUTPUT_DIR = "/home/adamm/Documents/FSOD/Data/Lavyanut_partial/Obj_Embs/"

# The number of images to be used from the few shots novel class.
SHOTS = 20

# Define Target Directories
TRAIN_BASE_DIR = os.path.join(BASE_OUTPUT_DIR, "train_cementmixertrucks/base_class")
TRAIN_CEMENT_DIR = os.path.join(BASE_OUTPUT_DIR, f"train_cementmixertrucks/cementmixertrucks_{SHOTS}_shots")

TEST_BASE_DIR = os.path.join(BASE_OUTPUT_DIR, "test_cementmixertrucks/base_class")
TEST_CEMENT_DIR = os.path.join(BASE_OUTPUT_DIR, f"test_cementmixertrucks/novel_class_cementmixertrucks_{SHOTS}_shots")

# Sanitized Novel Class Name (after spaces, commas, and hyphens are removed)
NOVEL_CEMENT = "CementMixerTrucks"

# Excluded classes sanitized
EXCLUDED_CLASSES =[
    "MobileCranes", 
    "ExtremelyLongHeavyDuty", 
    "Traileronly", 
    "HeavyDutyTractorTruck"
]

random.seed(42) # For reproducible splits

import os
import re

def sanitize_filenames(directory):
    
    # 2. Check if the directory even exists
    if not os.path.exists(directory):
        print("ERROR: That directory does not exist! Please check your path.")
        return

    # 3. Get ALL files in that folder
    all_items = os.listdir(directory)
    print(f"Found {len(all_items)} total items in this folder.")
    
    renamed_count = 0
    
    for filename in all_items:
        # 4. Only process specific file types (ignoring folders or hidden files)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.npy', '.txt')):
            continue
            
        filepath = os.path.join(directory, filename)
        name, ext = os.path.splitext(filename)
        
        # Replace spaces, commas, and hyphens with an underscore
        sanitized_name = re.sub(r'[ ,\-]', '_', name)
        
        # Replace multiple adjacent underscores with a single underscore
        sanitized_name = re.sub(r'_+', '_', sanitized_name) 
        
        new_filepath = os.path.join(directory, sanitized_name + ext)
        
        if filepath != new_filepath:
            os.rename(filepath, new_filepath)
            renamed_count += 1
            
    print(f"Successfully sanitized {renamed_count} files.\n")


# ================= 2. Split Function =================
def split_dataset():
    # Create all target directories
    for d in[TRAIN_BASE_DIR, TRAIN_CEMENT_DIR, 
              TEST_BASE_DIR, TEST_CEMENT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Dictionary to group files by image.
    image_catalog = {}
    
    files = glob.glob(os.path.join(SOURCE_DIR, '*.npy'))
    total_base_objects = 0
    
    print("Parsing files and grouping by image...")
    for filepath in files:
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        
        match = re.search(r'^(.*)_([a-zA-Z\-_]+)_(\d+)$', name_without_ext)
        
        if not match:
            print(f"Could not parse, skipping: {filename}")
            continue
            
        image_name = match.group(1) 
        class_name = match.group(2) 

        if class_name in EXCLUDED_CLASSES:
            continue 
            
        # Initialize dictionary for the image if it doesn't exist yet
        if image_name not in image_catalog:
            image_catalog[image_name] = {'cement': [], 'base':[]}
            
        if class_name == NOVEL_CEMENT:
            image_catalog[image_name]['cement'].append(filepath)
        else:
            image_catalog[image_name]['base'].append(filepath)
            total_base_objects += 1

    # --- Step A: Select Images for the Train Set ---
    train_images = set()
    test_images = set()
    
    train_cement_cnt, train_base_cnt = 0, 0
    target_base_train = int(0.8 * total_base_objects)

    cement_images = [img for img, data in image_catalog.items() if len(data['cement']) > 0]

    # Helper function to update counters
    def update_train_counters(img_name):
        nonlocal train_cement_cnt, train_base_cnt
        train_cement_cnt += len(image_catalog[img_name]['cement'])
        train_base_cnt += len(image_catalog[img_name]['base'])

    # 1. Grab images containing Cement Mixer Trucks
    for img in cement_images:
        if train_cement_cnt < 20: # Ensure we grab enough images to comfortably provide 'SHOTS'
            if img not in train_images:
                unassigned_cements =[i for i in cement_images if i not in train_images]
                
                # Leave at least one image with the novel class for the testing set
                if len(unassigned_cements) <= 1: break

                train_images.add(img)
                update_train_counters(img)

    # 2. Fill the rest of the Train set with Base images
    remaining_images =[img for img in image_catalog.keys() if img not in train_images]
    random.shuffle(remaining_images)
    
    for img in remaining_images:
        # STRICT PROTECTION: Any remaining image containing novel objects MUST go to test
        if len(image_catalog[img]['cement']) > 0:
            test_images.add(img)
            continue
            
        if train_base_cnt < target_base_train:
            train_images.add(img)
            train_base_cnt += len(image_catalog[img]['base'])
        else:
            test_images.add(img) 

    # --- Step B: Copy Files to Destinations ---
    print("Copying files to train and test directories...")
    copied_train_cement = 0
    
    for img, data in image_catalog.items():
        if img in train_images:
            # Copy exactly SHOTS
            for f in data['cement']:
                if copied_train_cement < SHOTS:
                    shutil.copy(f, TRAIN_CEMENT_DIR)
                    copied_train_cement += 1
                    
            # Copy base
            for f in data['base']:
                shutil.copy(f, TRAIN_BASE_DIR)

        elif img in test_images:
            for f in data['cement']:
                shutil.copy(f, TEST_CEMENT_DIR)
            for f in data['base']:
                shutil.copy(f, TEST_BASE_DIR)

    print("Done! Dataset successfully split at the image-level without data leakage.")
    print(f"  - Train Cement Mixer Trucks: {copied_train_cement} objects")
    print(f"  - Train Base: ~80% of data")
    print(f"  - Test data isolated to unique images.")

import os
import glob
import re

def print_actual_class_names(directory):
    # Set to store unique class names
    unique_classes = set()
    
    # Get all .npy files in the folder
    search_path = os.path.join(directory, "*.npy")
    files = glob.glob(search_path)
    
    # Regex explanation:
    # _([a-zA-Z\-_]+) : Matches an underscore, then captures 1 or more letters, hyphens, or underscores (the class name)
    # _(\d+)          : Matches an underscore, then captures 1 or more digits (the object ID)
    # (?:\.npy)?$     : Matches the end of the string, with an optional .npy extension
    regex = r'_([a-zA-Z\-_]+)_(\d+)(?:\.npy)?$'
    
    print("Scanning files to extract class names...\n")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        match = re.search(regex, filename)
        
        if match:
            class_name = match.group(1)
            unique_classes.add(class_name)
        else:
            print(f"[Warning] Could not parse class name from: {filename}")

    # Print the results alphabetically
    print(f"Found {len(unique_classes)} unique class names in the folder:")
    print("=" * 60)
    for cls in sorted(unique_classes):
        print(cls)



# Exact mapping from the old problematic names to the clean continuous names
CLASS_RENAME_MAP = {
    "Bulldozers": "Bulldozers",
    "CementMixerTrucks": "CementMixerTrucks",
    "ExtremelyLongHeavy-Duty": "ExtremelyLongHeavyDuty",
    "ExtremelyLongHeavy-Duty_Traileronly": "ExtremelyLongHeavyDutyTraileronly",
    "Forklifts": "Forklifts",
    "Heavy-Duty": "HeavyDuty",
    "LongHeavy-Duty": "LongHeavyDuty",
    "Medium-Small": "MediumSmall",
    "Medium-Standard": "MediumStandard",
    "MobileCranes": "MobileCranes",
    "Other": "Other",
    "Small": "Small",
    "TruckTractor": "TruckTractor"
}

def rename_classes_in_files():
    print("Renaming files to clean up class names...")
    files = sorted(glob.glob(os.path.join(SOURCE_DIR, '*.npy')))
    renamed_count = 0
    
    for filepath in files:
        dirname, filename = os.path.split(filepath)
        
        # Check every old class in our mapping
        for old_class, new_class in CLASS_RENAME_MAP.items():
            if old_class == new_class:
                continue # Skip if there's no change needed
                
            # Regex pattern to match exactly: _{OldClassName}_{Digits}.npy at the end of the file
            # This ensures we don't accidentally replace text inside the image coordinates
            pattern = rf'_({re.escape(old_class)})_(\d+)\.npy$'
            
            if re.search(pattern, filename):
                # Replace the old class name with the new clean one
                new_filename = re.sub(pattern, rf'_{new_class}_\2.npy', filename)
                new_filepath = os.path.join(dirname, new_filename)
                
                # Rename the actual file
                os.rename(filepath, new_filepath)
                renamed_count += 1
                break # Move to the next file once renamed

    print(f"Successfully renamed {renamed_count} files!")


def find_mismatched_labels(label_dir, image_dir):
    # Get sets of filenames without their extensions
    labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}
    images = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}

    # Find labels that exist in the labels set but not the images set
    mismatched = labels - images

    if mismatched:
        print(f"Found {len(mismatched)} orphan labels:")
        for name in sorted(mismatched):
            print(name)
    else:
        print("No mismatched labels found.")


def count_crop_classes(crops_dir, output_txt_path):
    # Use a Counter to automatically tally up class names
    class_counts = Counter()
    
    # Get all files in the directory
    files = glob.glob(os.path.join(crops_dir, '*.*'))
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    for filepath in files:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        
        # Only process image files
        if ext.lower() not in valid_extensions:
            continue
            
        # Split the filename by underscores
        parts = name.split('_')
        
        # Assuming format: ..._04.2024_MediumSmall_1710
        # The class name is the second-to-last part (index -2)
        if len(parts) >= 2:
            class_name = parts[-2]
            class_counts[class_name] += 1
            
    # Save the tallies to a text file
    with open(output_txt_path, 'w') as f:
        f.write("Dataset Class Counts:\n")
        f.write("-" * 25 + "\n")
        
        # .most_common() orders them from highest count to lowest count
        for cls_name, count in class_counts.most_common():
            f.write(f"{cls_name}: {count}\n")
            
    print(f"Successfully counted {sum(class_counts.values())} objects across {len(class_counts)} classes.")
    print(f"Results saved to: {output_txt_path}")

if __name__ == "__main__":
    # label_path = "/home/adamm/Documents/FSOD/Data/Lavyanut/labels"
    # image_path = "/home/adamm/Documents/FSOD/Data/Lavyanut/images"

    # Run the function
    # find_mismatched_labels(label_path, image_path)
    # sanitize_filenames(SOURCE_DIR)
    # rename_classes_in_files()
    # print_actual_class_names(TRAIN_BASE_DIR)
    # print_actual_class_names(TRAIN_FORK_DIR)
    # print_actual_class_names(TRAIN_TRAIL_DIR)
    # print_actual_class_names(TEST_BASE_DIR)
    # print_actual_class_names(TEST_FORK_DIR)
    # print_actual_class_names(TEST_TRAIL_DIR)
    # print_actual_class_names(TEST_HEAVY_DIR)

    # crops_directory = "/home/adamm/Documents/FSOD/Data/Lavyanut/Obj_Crops/"
    # output_file = "/home/adamm/Documents/FSOD/Data/Lavyanut/class_counts.txt"

    # count_crop_classes(crops_directory, output_file)
    split_dataset()