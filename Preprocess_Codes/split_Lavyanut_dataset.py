import os
import glob
import re
import shutil
import random

# ================= Paths & Settings =================
SOURCE_DIR = "/home/adamm/Documents/FSOD/Data/Lavyanut/Obj_Embs/All_Embs/"
BASE_OUTPUT_DIR = "/home/adamm/Documents/FSOD/Data/Lavyanut/Obj_Embs/"

# Define Target Directories
TRAIN_BASE_DIR = os.path.join(BASE_OUTPUT_DIR, "train/base_class")
TRAIN_FORK_DIR = os.path.join(BASE_OUTPUT_DIR, "train/forklifts_few_shots")
TRAIN_TRAIL_DIR = os.path.join(BASE_OUTPUT_DIR, "train/trailer_few_shots")

TEST_BASE_DIR = os.path.join(BASE_OUTPUT_DIR, "test/base_class")
TEST_FORK_DIR = os.path.join(BASE_OUTPUT_DIR, "test/novel_class_forklifts")
TEST_TRAIL_DIR = os.path.join(BASE_OUTPUT_DIR, "test/novel_class_trailer")

# Sanitized Novel Class Names (after spaces, commas, and hyphens are removed)
NOVEL_FORKLIFT = "Forklifts"
NOVEL_TRAILER = "ExtremelyLongHeavyDutyTraileronly"

# Previously mentioned excluded classes (<100 objects) sanitized
EXCLUDED_CLASSES =["ExtremelyLongHeavyDuty", "HeavyDutyTractorTruck", "MobileCranes"]

random.seed(42) # For reproducible splits

# ================= 1. Sanitization Function =================
def sanitize_filenames(directory):
    """
    Removes spaces, commas, and hyphens from all NPY filenames in the directory
    so it becomes a continuous string separated only by underscores.
    """
    print("Sanitizing filenames...")
    files = glob.glob(os.path.join(directory, '*.npy'))
    
    renamed_count = 0
    for filepath in files:
        dirname, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)
        
        # Remove spaces, commas, and hyphens
        sanitized_name = re.sub(r'[ ,]', '_', name)
        
        new_filepath = os.path.join(dirname, sanitized_name + ext)
        if filepath != new_filepath:
            os.rename(filepath, new_filepath)
            renamed_count += 1
            
    print(f"Sanitized {renamed_count} files.\n")

# ================= 2. Split Function =================
def split_dataset():
    # Create all target directories
    for d in[TRAIN_BASE_DIR, TRAIN_FORK_DIR, TRAIN_TRAIL_DIR, TEST_BASE_DIR, TEST_FORK_DIR, TEST_TRAIL_DIR]:
        os.makedirs(d, exist_ok=True)

    # Dictionary to group files by image. Format:
    # { image_name: {'forklifts': [], 'trailers': [], 'base':[]} }
    image_catalog = {}
    
    files = glob.glob(os.path.join(SOURCE_DIR, '*.npy'))
    total_base_objects = 0
    
    print("Parsing files and grouping by image...")
    for filepath in files:
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        
        # We use Regex to safely split the name.
        # It looks for: (image_name) _ (class_name_with_letters_and_hyphens) _ (object_id_digits)
        match = re.search(r'^(.*)_([a-zA-Z\-_]+)_(\d+)$', name_without_ext)
        

        
        if not match:
            print(f"Could not parse, skipping: {filename}")
            continue
            
        image_name = match.group(1)  # e.g., '50_Or_003_35.654030__-0.565647_03.2021'
        class_name = match.group(2)  # e.g., 'ExtremelyLongHeavy-Duty_Traileronly'

        if "orklift" in class_name: # Case-insensitive partial match for debugging
            print(f"Found a forklift file! Class parsed as: '{class_name}'")
        
        if class_name in EXCLUDED_CLASSES:
            continue # Skip excluded classes entirely
            
        if image_name not in image_catalog:
            image_catalog[image_name] = {'forklifts':[], 'trailers':[], 'base':[]}
            
        if class_name == NOVEL_FORKLIFT:
            image_catalog[image_name]['forklifts'].append(filepath)
        elif class_name == NOVEL_TRAILER:
            image_catalog[image_name]['trailers'].append(filepath)
        else:
            image_catalog[image_name]['base'].append(filepath)
            total_base_objects += 1

    # --- Step A: Select Images for the Train Set ---
    train_images = set()
    test_images = set()
    
    train_fork_cnt, train_trail_cnt, train_base_cnt = 0, 0, 0
    target_base_train = int(0.8 * total_base_objects)

    # Pre-identify all images containing the novel classes
    fork_images = [img for img, data in image_catalog.items() if len(data['forklifts']) > 0]
    trail_images =[img for img, data in image_catalog.items() if len(data['trailers']) > 0]

    # 1. Grab images containing Forklifts until we reach >= 20 objects
    for img in fork_images:
        if train_fork_cnt < 20:
            if img not in train_images:
                unassigned_forks =[i for i in fork_images if i not in train_images]
                unassigned_trails =[i for i in trail_images if i not in train_images]
                
                # Stop taking forklifts if this is the very last image for the test set
                if len(unassigned_forks) <= 1:
                    break
                
                # Skip this image if it's the very last trailer image for the test set
                if img in trail_images and len(unassigned_trails) <= 1:
                    continue

                train_images.add(img)
                train_fork_cnt += len(image_catalog[img]['forklifts'])
                train_trail_cnt += len(image_catalog[img]['trailers'])
                train_base_cnt += len(image_catalog[img]['base'])

    # 2. Grab images containing Trailers until we reach >= 20 objects
    for img in trail_images:
        if train_trail_cnt < 20:
            if img not in train_images:
                unassigned_trails = [i for i in trail_images if i not in train_images]
                unassigned_forks = [i for i in fork_images if i not in train_images]
                
                # Stop taking trailers if this is the very last image for the test set
                if len(unassigned_trails) <= 1:
                    break
                
                # Skip this image if it's the very last forklift image for the test set
                if img in fork_images and len(unassigned_forks) <= 1:
                    continue

                train_images.add(img)
                train_fork_cnt += len(image_catalog[img]['forklifts'])
                train_trail_cnt += len(image_catalog[img]['trailers'])
                train_base_cnt += len(image_catalog[img]['base'])

    # 3. Fill the rest of the Train set with Base images up to the 80% mark
    remaining_images = [img for img in image_catalog.keys() if img not in train_images]
    random.shuffle(remaining_images)
    
    for img in remaining_images:
        # STRICT PROTECTION: If an image was deliberately saved because it contains 
        # a novel class, force it to the test set so the base-filler loop doesn't steal it!
        if len(image_catalog[img]['forklifts']) > 0 or len(image_catalog[img]['trailers']) > 0:
            test_images.add(img)
            continue
            
        if train_base_cnt < target_base_train:
            train_images.add(img)
            train_base_cnt += len(image_catalog[img]['base'])
        else:
            test_images.add(img) # The 20% remainder goes to test

    # --- Step B: Copy Files to Destinations ---
    print("Copying files to train and test directories...")
    copied_train_fork, copied_train_trail = 0, 0
    
    for img, data in image_catalog.items():
        if img in train_images:
            # Copy exactly 20 Forklifts to Train
            for f in data['forklifts']:
                if copied_train_fork < 20:
                    shutil.copy(f, TRAIN_FORK_DIR)
                    copied_train_fork += 1
            
            # Copy exactly 20 Trailers to Train
            for f in data['trailers']:
                if copied_train_trail < 20:
                    shutil.copy(f, TRAIN_TRAIL_DIR)
                    copied_train_trail += 1
                    
            # Copy all base objects from this image to Train
            for f in data['base']:
                shutil.copy(f, TRAIN_BASE_DIR)
                
            # NOTE: If an image had a 21st Forklift, it is ignored here. 
            # It is NOT copied to Train (maintaining your 20-shot rule) and 
            # NOT copied to Test (preventing image-background data leakage).

        elif img in test_images:
            # Everything in the test images goes to the test folders
            for f in data['forklifts']:
                shutil.copy(f, TEST_FORK_DIR)
            for f in data['trailers']:
                shutil.copy(f, TEST_TRAIL_DIR)
            for f in data['base']:
                shutil.copy(f, TEST_BASE_DIR)

    print("Done! Dataset successfully split at the image-level without data leakage.")
    print(f"  - Train Forklifts: {copied_train_fork} objects")
    print(f"  - Train Trailers: {copied_train_trail} objects")
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


SOURCE_DIR = "/home/adamm/Documents/FSOD/Data/Lavyanut/Obj_Embs/All_Embs/"

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
    files = glob.glob(os.path.join(SOURCE_DIR, '*.npy'))
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

if __name__ == "__main__":
    # sanitize_filenames(SOURCE_DIR)
    # rename_classes_in_files()
    # print_actual_class_names(TRAIN_BASE_DIR)
    # print_actual_class_names(TRAIN_FORK_DIR)
    # print_actual_class_names(TRAIN_TRAIL_DIR)
    # print_actual_class_names(TEST_BASE_DIR)
    # print_actual_class_names(TEST_FORK_DIR)
    # print_actual_class_names(TEST_TRAIL_DIR)
    split_dataset()