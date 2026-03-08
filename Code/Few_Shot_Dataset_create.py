import os
import shutil
import random

# ============================
# PATHS
# ============================

root_dir = "/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val"

novel_dir = os.path.join(root_dir, "novel_classes")
base_dir = os.path.join(root_dir, "base_classes")

few_shots_root = os.path.join(root_dir, "splits","few_shots")
inference_root = os.path.join(root_dir, "splits", "inference_objs")

shot_levels = [30, 20, 10, 5]
NUM_BASE_INFERENCE = 4000

# ============================
# CREATE DIRECTORIES
# ============================

for shot in shot_levels:
    os.makedirs(os.path.join(few_shots_root, str(shot)), exist_ok=True)

os.makedirs(inference_root, exist_ok=True)

# ============================
# LOAD FILES
# ============================

novel_files = [f for f in os.listdir(novel_dir) if f.endswith(".npy")]
base_files = [f for f in os.listdir(base_dir) if f.endswith(".npy")]

if len(novel_files) < 30:
    raise ValueError("Not enough novel class files to sample 30.")

if len(base_files) < NUM_BASE_INFERENCE:
    raise ValueError("Not enough base class files to sample 4000.")

# Optional reproducibility
random.seed(42)

# ============================
# NOVEL FEW-SHOT SELECTION
# ============================

selected_30 = random.sample(novel_files, 30)

for f in selected_30:
    shutil.copy2(os.path.join(novel_dir, f),
                 os.path.join(few_shots_root, "30", f))

selected_20 = random.sample(selected_30, 20)
for f in selected_20:
    shutil.copy2(os.path.join(novel_dir, f),
                 os.path.join(few_shots_root, "20", f))

selected_10 = random.sample(selected_20, 10)
for f in selected_10:
    shutil.copy2(os.path.join(novel_dir, f),
                 os.path.join(few_shots_root, "10", f))

selected_5 = random.sample(selected_10, 5)
for f in selected_5:
    shutil.copy2(os.path.join(novel_dir, f),
                 os.path.join(few_shots_root, "5", f))

# Remaining novel files → inference
remaining_novel = list(set(novel_files) - set(selected_30))

for f in remaining_novel:
    shutil.copy2(os.path.join(novel_dir, f),
                 os.path.join(inference_root, f))

# ============================
# BASE → INFERENCE (4000 RANDOM)
# ============================

selected_base = random.sample(base_files, NUM_BASE_INFERENCE)

for f in selected_base:
    shutil.copy2(os.path.join(base_dir, f),
                 os.path.join(inference_root, f))

# ============================
# SUMMARY
# ============================

print("Few-shot splits created:")
print(f"30-shot: {len(selected_30)}")
print(f"20-shot: {len(selected_20)}")
print(f"10-shot: {len(selected_10)}")
print(f"5-shot:  {len(selected_5)}")

print("\nInference set:")
print(f"Novel objects: {len(remaining_novel)}")
print(f"Base objects:  {len(selected_base)}")
print(f"Total inference: {len(remaining_novel) + len(selected_base)}")