import os
import shutil
from collections import Counter

def count_objects_by_class(crops_dir, output_file="class_counts.txt"):
    class_counts = Counter()

    for filename in os.listdir(crops_dir):
        if not os.path.isfile(os.path.join(crops_dir, filename)):
            continue

        name = os.path.splitext(filename)[0]
        obj_class = name.rsplit("_", 1)[0].split("_")[-1]
        class_counts[obj_class] += 1

    with open(output_file, "w") as f:
        for cls, count in sorted(class_counts.items()):
            f.write(f"{cls}: {count}\n")

    print(f"Saved counts to: {os.path.abspath(output_file)}")

def organize_crops_by_class(base_dir):
    crops_dir = os.path.join(base_dir, "Obj_Crops")
    output_dir = os.path.join(base_dir, "Obj_Crops_Organized")

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(crops_dir):
        src_path = os.path.join(crops_dir, filename)

        if not os.path.isfile(src_path):
            continue

        name = os.path.splitext(filename)[0]
        obj_class = name.rsplit("_", 1)[0].split("_")[-1]

        class_dir = os.path.join(output_dir, obj_class)
        os.makedirs(class_dir, exist_ok=True)

        shutil.copy2(src_path, os.path.join(class_dir, filename))

    print(f"Organized crops saved in: {output_dir}")

# Example
crops_dir = "/home/adamm/Documents/FSOD/Data/Lavyanut/Images/train/Obj_Crops/"
# count_objects_by_class(crops_dir)
base_dir = "/home/adamm/Documents/FSOD/Data/Lavyanut/Images/train/"
organize_crops_by_class(base_dir)