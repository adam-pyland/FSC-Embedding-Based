import os
import glob
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_npy_folder(folder_path):
    file_paths = glob.glob(os.path.join(folder_path, "*.npy"))
    if not file_paths:
        raise ValueError(f"No .npy files found in {folder_path}")
    
    features = []
    class_names = []

    for f in file_paths:
        features.append(np.load(f))

        filename = os.path.basename(f)
        parts = filename.split("_")
        class_name = parts[-2]
        class_names.append(class_name)
    
    return np.vstack(features), class_name

def generate_smote_features(original_features, num_to_generate, k_neighbors=5, seed=42):
    rng = np.random.default_rng(seed)
    n_existing = original_features.shape[0]
    k = min(k_neighbors, n_existing - 1)
    
    nn = NearestNeighbors(n_neighbors=k+1).fit(original_features)
    distances, indices = nn.kneighbors(original_features)
    
    synthetic_samples = []
    
    for _ in range(num_to_generate):
        base_idx = rng.integers(0, n_existing)
        base_point = original_features[base_idx]
        
        neighbor_idx = rng.choice(indices[base_idx, 1:])
        neighbor_point = original_features[neighbor_idx]
        
        gap = rng.random()
        new_point = base_point + gap * (neighbor_point - base_point)
        
        synthetic_samples.append(new_point)
        
    synthetic_samples = np.array(synthetic_samples)
    
    norms = np.linalg.norm(synthetic_samples, axis=1, keepdims=True)
    synthetic_samples = synthetic_samples / (norms + 1e-8)
    
    return synthetic_samples

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    WORK_PLACE = 'yehud' # The place where I am working in: 'yehud' or 'matrix'. Or WSL if decided to work on WSL on windows in Yehud.

    data_path = r'C:\Adams\FSOD\Data\Lavyanut\Lavyanut' if WORK_PLACE is 'yehud' else '/home/adamm/Documents/FSOD/Data/Lavyanut'

    SHOTS = 20
    TARGET_CLASS = 'trailer'
    few_shots_folder = fr"{data_path}/Obj_Embs/train/{TARGET_CLASS}_{SHOTS}_shots"
    output_folder = fr"{data_path}/Obj_Embs/train/Generated_{TARGET_CLASS}_{SHOTS}_shots"
    
    os.makedirs(output_folder, exist_ok=True)
    
    original_shots, class_name = load_npy_folder(few_shots_folder)
    print(f"Detected class: {class_name}")
    print(f"Original shots shape: {original_shots.shape}")
    
    synthetic_shots = generate_smote_features(
        original_features=original_shots, 
        num_to_generate=80, 
        k_neighbors=5
    )
    print(f"Synthetic shots shape: {synthetic_shots.shape}")
    
    # NEW: save each synthetic sample
    for i, sample in enumerate(synthetic_shots):
        save_path = os.path.join(output_folder, f"synthetic_{class_name}_{i:03d}.npy")
        np.save(save_path, sample)
    
    final_novel_class_features = np.vstack((original_shots, synthetic_shots))
    print(f"Final training set shape: {final_novel_class_features.shape}")