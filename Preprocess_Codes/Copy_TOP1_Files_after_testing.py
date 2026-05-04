import os
import shutil

def copy_class_files(txt_file_path, source_folder, class_name, destination_folder, additional_folder, test_destination_folder=None):
    # Ensure the destination folders exist
    os.makedirs(destination_folder, exist_ok=True)
    if test_destination_folder:
        os.makedirs(test_destination_folder, exist_ok=True)
    
    # 1. Read the text file and store names in a SET for fast lookup
    with open(txt_file_path, 'r') as f:
        # Strip whitespace/newlines and ignore empty lines
        txt_files_set = {line.strip() for line in f if line.strip()}
        
    # 2. Copy matching files from the text list to the destination folder (train)
    for file_name in txt_files_set:
        # Check if the class name is exactly matched in the file name
        if f"_{class_name}_" in file_name:
            src_path = os.path.join(source_folder, file_name)
            dst_path = os.path.join(destination_folder, file_name)
            
            # Copy if the file actually exists in the source folder
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Missing file in source folder: {file_name}")

    # 3. NEW: Copy the REST of the files from source_folder to test_destination_folder
    if test_destination_folder and os.path.exists(source_folder):
        for file_name in os.listdir(source_folder):
            src_path = os.path.join(source_folder, file_name)
            
            # If it's a file, ends with .npy, and is NOT in the text file list
            if os.path.isfile(src_path) and file_name.endswith('.npy') and file_name not in txt_files_set:
                dst_path = os.path.join(test_destination_folder, file_name)
                shutil.copy2(src_path, dst_path)

    # 4. Copy ALL .npy files from the additional folder to destination_folder (train)
    if os.path.exists(additional_folder):
        for file_name in os.listdir(additional_folder):
            if file_name.endswith('.npy'):
                src_path = os.path.join(additional_folder, file_name)
                dst_path = os.path.join(destination_folder, file_name)
                
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
    else:
        print(f"Warning: Additional folder not found: {additional_folder}")

# --- Example Usage ---
target_class = 'trailer'
SHOTS = 20
txt_path = r'C:\Adams\FSOD\Codes\FSC-Embedding-Based\Outputs_Old_Dataset\Outputs_Generalized_Windows\Lavyanut\20_shots\ExtremelyLongHeavyDutyTraileronly\TopK_Evaluation_Lists-focal_center-Loss-cosine-Logits-Based\ExtremelyLongHeavyDutyTraileronly_TOP1_List.txt'

src_dir = fr'C:\Adams\FSOD\Data\Lavyanut\Lavyanut_old\Obj_Embs\test\novel_class_{target_class}_{SHOTS}_shots'
extra_dir = fr'C:\Adams\FSOD\Data\Lavyanut\Lavyanut_old\Obj_Embs\train\{target_class}_{SHOTS}_shots'

dest_dir = fr'C:\Adams\FSOD\Data\Lavyanut\Lavyanut_old\Obj_Embs\train\novel_class_TOP1_{target_class}_{SHOTS}_shots'
test_dest_dir = fr'C:\Adams\FSOD\Data\Lavyanut\Lavyanut_old\Obj_Embs\test\novel_class_TOP1_{target_class}_{SHOTS}_shots'

# Ensure class name strictly matches the format in the file strings
if target_class == 'trailer':
    target_class = 'ExtremelyLongHeavyDutyTraileronly'

# Run the function
copy_class_files(
    txt_file_path=txt_path, 
    source_folder=src_dir, 
    class_name=target_class, 
    destination_folder=dest_dir, 
    additional_folder=extra_dir, 
    test_destination_folder=test_dest_dir
)