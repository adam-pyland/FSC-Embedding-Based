import os
from glob import glob

dirs = [
    "/home/adamm/Documents/FSOD/Data/Lavyanut_partial/Obj_Embs/test/base_class/",
    "/home/adamm/Documents/FSOD/Data/Lavyanut_partial/Obj_Embs/test/novel_class_trailer_20_shots/"
]

image_names = set()

for d in dirs:
    npy_files = glob(os.path.join(d, "*.npy"))

    for f in npy_files:
        filename = os.path.basename(f)

        # Split from the right:
        # image_name__class_name_objectid.npy
        left_part = filename.rsplit("_", 1)[0]   # remove object id
        image_name = left_part.split("__")[0]    # keep image name only

        image_names.add(image_name)

print(f"Number of unique images: {len(image_names)}")