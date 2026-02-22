import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
print("Working directory:", os.getcwd())

def visualize_tsne(root_feature_dir, output_plot='Outputs/T-SNE/tsne_Base_vs_Novel_Unified.png'):
    """
    root_feature_dir: The parent folder containing 'base_classes' and 'novel_classes' subfolders.
    """
    
    # ==========================================
    # 1. CONFIGURATION: PATHS AND CLASSES
    # ==========================================
    
    # Define the two unified groups and their source folders
    # Structure: (Subfolder Name, List of Specific Classes, Unified Label Name)
    
    base_group_config = {
        'folder': 'base_classes',
        'targets': [
            'Small Car', 'Bus', 'Dump Truck', 'Van', 'Tractor', 
            'Truck Tractor', 'Excavator', 'other-vehicle'
        ],
        'label': 'Base Classes' # Unified Name
    }

    novel_group_config = {
        'folder': 'novel_classes',
        'targets': ['Cargo Truck', 'Trailer'],
        'label': 'Novel Classes' # Unified Name
    }
    
    groups_to_process = [base_group_config, novel_group_config]

    # ==========================================
    
    features = []
    labels = []

    # Helper function to match filenames
    def match_class(name_string, target_list):
        for cls in target_list:
            # Check exact match OR match with spaces replaced by underscores
            cls_underscore = cls.replace(' ', '_')
            # Check if filename ends with class name (ignoring the count and extension)
            if name_string.endswith(cls) or name_string.endswith(cls_underscore):
                return True
        return False

    # 2. LOAD DATA FROM BOTH DIRECTORIES
    print(f"Scanning root directory: {root_feature_dir}")
    
    for group in groups_to_process:
        folder_path = os.path.join(root_feature_dir, group['folder'])
        target_list = group['targets']
        unified_label = group['label']
        
        print(f"Processing group: {unified_label} from {folder_path}...")
        
        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} does not exist. Skipping.")
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        count = 0
        for f in tqdm(files, desc=unified_label):
            # Parse filename: "v_0_Small_Car_1.npy" -> "v_0_Small_Car"
            name_no_ext = os.path.splitext(f)[0] 
            name_no_count = name_no_ext.rsplit('_', 1)[0]
            
            # Verify this file actually belongs to the target list for this folder
            if match_class(name_no_count, target_list):
                feat = np.load(os.path.join(folder_path, f))
                features.append(feat)
                labels.append(unified_label) # Assign the Unified Label
                count += 1
        
        print(f"  -> Loaded {count} samples for {unified_label}")

    X = np.array(features)
    y = np.array(labels)

    if len(X) == 0:
        print("No matching classes found in any directory.")
        return

    # 3. COMPUTE T-SNE
    print(f"Computing T-SNE on {len(X)} total samples...")
    unique_labels = np.unique(y)
    print(f"Visualizing groups: {unique_labels}")

    # Adjust perplexity based on sample size
    n_samples = len(X)
    perp = 30 if n_samples > 40 else max(5, n_samples // 2)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto', verbose=1)
    X_embedded = tsne.fit_transform(X)

    # 4. PLOTTING
    plt.figure(figsize=(12, 10))
    
    # Create specific colors: Blue for Base, Orange/Red for Novel
    palette = sns.color_palette("bright", n_colors=len(unique_labels))
    
    # Main Scatter Plot
    sns.scatterplot(
        x=X_embedded[:,0], y=X_embedded[:,1], 
        hue=y, style=y, 
        palette=palette, 
        s=80, alpha=0.6
    )
    
    # --- Draw Center Points (Centroids) & Labels ---
    for label in unique_labels:
        # Get indices for the current unified group
        indices = np.where(y == label)
        class_points = X_embedded[indices]
        
        # Calculate Centroid
        centroid = np.mean(class_points, axis=0)
        
        # 1. Plot Big Red Dot at Center
        plt.scatter(centroid[0], centroid[1], c='red', s=250, marker='o', 
                    edgecolors='black', linewidths=2, zorder=10, label='_nolegend_')

        # 2. Write the Unified Group Name
        plt.text(centroid[0], centroid[1], label, fontsize=12, fontweight='bold', 
                 color='black', ha='center', va='center', zorder=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.title("T-SNE: Base Classes vs. Novel Classes", fontsize=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    plt.legend(title="Unified Groups", loc='upper right')
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    # Point to the PARENT directory containing 'base_classes' and 'novel_classes'
    input_root = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/DINO_FAIR1M_features'
    
    if os.path.exists(input_root):
        visualize_tsne(input_root)
    else:
        print(f"Please check the input path: {input_root}")