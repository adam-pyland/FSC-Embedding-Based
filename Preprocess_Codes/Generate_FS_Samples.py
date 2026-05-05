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


def generate_smote_features_v2(original_features, num_to_generate, k_neighbors=5, seed=42):
    rng = np.random.default_rng(seed)  # ✅ seeded RNG

    n_existing = original_features.shape[0]
    k = min(k_neighbors, n_existing - 1)

    nn = NearestNeighbors(n_neighbors=k+1).fit(original_features)
    distances, indices = nn.kneighbors(original_features)

    # --- compute "difficulty" score ---
    difficulty = distances[:, 1:].mean(axis=1)
    prob_base = difficulty / difficulty.sum()

    synthetic_samples = []

    for _ in range(num_to_generate):
        # --- sample base using difficulty ---
        base_idx = rng.choice(n_existing, p=prob_base)
        base_point = original_features[base_idx]

        # --- bias toward farther neighbors ---
        neighbor_distances = distances[base_idx, 1:]
        prob_neighbors = neighbor_distances / neighbor_distances.sum()

        neighbor_idx = rng.choice(indices[base_idx, 1:], p=prob_neighbors)
        neighbor_point = original_features[neighbor_idx]

        gap = rng.random()  # ✅ seeded gap in [0,1)
        new_point = base_point + gap * (neighbor_point - base_point)

        synthetic_samples.append(new_point)

    synthetic_samples = np.array(synthetic_samples)

    # Normalize (same as before)
    norms = np.linalg.norm(synthetic_samples, axis=1, keepdims=True)
    synthetic_samples = synthetic_samples / (norms + 1e-8)

    return synthetic_samples

def slerp(a, b, t):
    """Spherical linear interpolation between vectors a and b"""
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-6:
        return a  # very close vectors

    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * a + \
           (np.sin(t * theta) / sin_theta) * b


def generate_smote_features_cosine(original_features, num_to_generate, k_neighbors=5, seed=42):
    rng = np.random.default_rng(seed)

    # Ensure normalized (important!)
    original_features = original_features / (
        np.linalg.norm(original_features, axis=1, keepdims=True) + 1e-8
    )

    n_existing = original_features.shape[0]
    k = min(k_neighbors, n_existing - 1)

    # Use cosine distance
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(original_features)
    distances, indices = nn.kneighbors(original_features)

    # --- difficulty (same idea, now cosine-based) ---
    difficulty = distances[:, 1:].mean(axis=1)
    if difficulty.sum() == 0:
        prob_base = np.ones(n_existing) / n_existing
    else:
        prob_base = difficulty / difficulty.sum()

    synthetic_samples = []

    for _ in range(num_to_generate):
        # 1. Sample base (harder points preferred)
        base_idx = rng.choice(n_existing, p=prob_base)
        base = original_features[base_idx]

        # 2. Sample neighbor (farther ones preferred)
        neighbor_distances = distances[base_idx, 1:]
        if neighbor_distances.sum() == 0:
            prob_neighbors = np.ones_like(neighbor_distances) / len(neighbor_distances)
        else:
            prob_neighbors = neighbor_distances / neighbor_distances.sum()

        neighbor_idx = rng.choice(indices[base_idx, 1:], p=prob_neighbors)
        neighbor = original_features[neighbor_idx]

        # 3. Interpolation factor
        t = rng.random()

        # 4. SLERP instead of linear interpolation
        new_point = slerp(base, neighbor, t)

        synthetic_samples.append(new_point)

    return np.array(synthetic_samples)

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
    
    synthetic_shots = generate_smote_features_cosine(
        original_features=original_shots, 
        num_to_generate=480, 
        k_neighbors=5
    )
    print(f"Synthetic shots shape: {synthetic_shots.shape}")
    
    # NEW: save each synthetic sample
    for i, sample in enumerate(synthetic_shots):
        save_path = os.path.join(output_folder, f"synthetic_{class_name}_{i:03d}.npy")
        np.save(save_path, sample)
    
    final_novel_class_features = np.vstack((original_shots, synthetic_shots))
    print(f"Final training set shape: {final_novel_class_features.shape}")