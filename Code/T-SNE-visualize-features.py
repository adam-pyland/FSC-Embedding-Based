import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm

def visualize_tsne(feature_dir, output_plot='tsne_plot.png'):
    # 1. Define classes (as formatted in your filenames: spaces -> underscores)
    target_classes = [
        'Small_Car', 'Bus', 'Dump_Truck', 'Van', 'Tractor', 
        'Truck_Tractor', 'Excavator', 'other-vehicle',
        'Cargo_Truck', 'Trailer'
    ]
    
    features = []
    labels = []

    # 2. Load Data and extract labels from filenames
    print("Loading features...")
    for f in tqdm(os.listdir(feature_dir)):
        if f.endswith('.npy'):
            # Load feature vector
            feat = np.load(os.path.join(feature_dir, f))
            features.append(feat)
            
            # Parse label: Filename is "ImgID_ClassName_Count.npy"
            # We strip extension, strip the counter at the end, and match the suffix
            name_no_ext = os.path.splitext(f)[0] # "v_0_Small_Car_1"
            name_no_count = name_no_ext.rsplit('_', 1)[0] # "v_0_Small_Car"
            
            # Find which class this file belongs to
            found_label = "Unknown"
            for cls in target_classes:
                if name_no_count.endswith(cls):
                    found_label = cls
                    break
            labels.append(found_label)

    X = np.array(features)
    y = np.array(labels)

    # 3. Compute T-SNE
    print(f"Computing T-SNE on {len(X)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto', verbose=1)
    X_embedded = tsne.fit_transform(X)

    # 4. Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, palette="tab10", s=60, alpha=0.8)
    
    plt.title("T-SNE Visualization of ResNet50 Features")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")
    plt.tight_layout()
    
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    # Point this to the 'individual_features' folder created by your feature extractor
    input_dir = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/Input_Features/individual_features'
    visualize_tsne(input_dir)