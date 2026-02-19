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

def visualize_tsne(feature_dir, output_plot='Outputs/T-SNE/tsne_Base_plot.png'):
    # 1. Define classes 
    # Note: The code below handles spaces vs underscores automatically
    target_classes = [
        'Small Car', 'Bus', 'Dump Truck', 'Van', 'Tractor', 
        'Truck Tractor', 'Excavator', 'other-vehicle'
    ]
    
    features = []
    labels = []

    # 2. Load Data and extract labels from filenames
    print("Loading features...")
    if not os.path.exists(feature_dir):
        print(f"Error: Directory {feature_dir} does not exist.")
        return

    for f in tqdm(os.listdir(feature_dir)):
        if f.endswith('.npy'):
            # Load feature vector
            feat = np.load(os.path.join(feature_dir, f))
            features.append(feat)
            
            # Parse label: Filename is "ImgID_ClassName_Count.npy"
            # Example: "v_0_Small_Car_1.npy" -> "v_0_Small_Car"
            name_no_ext = os.path.splitext(f)[0] 
            name_no_count = name_no_ext.rsplit('_', 1)[0] 
            
            # Find which class this file belongs to
            found_label = "Unknown"
            for cls in target_classes:
                # Check exact match OR match with spaces replaced by underscores
                # e.g., 'Small Car' will match '..._Small_Car'
                cls_underscore = cls.replace(' ', '_')
                
                if name_no_count.endswith(cls) or name_no_count.endswith(cls_underscore):
                    found_label = cls
                    break
            labels.append(found_label)

    if len(features) == 0:
        print("No .npy files found.")
        return

    X = np.array(features)
    y = np.array(labels)

    # 3. Compute T-SNE
    print(f"Computing T-SNE on {len(X)} samples with {len(np.unique(y))} unique classes...")
    # Perplexity lowered slightly to accommodate cases with fewer samples per class, can be adjusted
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto', verbose=1)
    X_embedded = tsne.fit_transform(X)

    # 4. Plot
    plt.figure(figsize=(14, 12)) # Made figure slightly larger for multiple classes
    
    # Draw the individual points
    # using 'tab10' or 'Set1' to ensure enough distinct colors for 8+ classes
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, palette="tab10", s=50, alpha=0.6)
    
    # --- Draw Center Points and Labels for ALL classes ---
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        if label == "Unknown": continue 

        # Get indices for the current class
        indices = np.where(y == label)
        
        # Get the embedded coordinates for this class
        class_points = X_embedded[indices]
        
        # Calculate the centroid (mean of x and y)
        centroid = np.mean(class_points, axis=0)
        
        # 1. Plot the BIG RED DOT
        plt.scatter(centroid[0], centroid[1], 
                    c='red',            
                    s=250,              # Big size
                    marker='o',         
                    edgecolors='black', 
                    linewidths=2,
                    zorder=10, 
                    label='_nolegend_') # Don't add red dots to legend

        # 2. Write the Class Name at the center
        plt.text(centroid[0], centroid[1], 
                 label, 
                 fontsize=10, 
                 fontweight='bold', 
                 color='black',
                 ha='center', 
                 va='center', 
                 zorder=11,
                 # Add a small box behind text to make it readable over the red dot
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7))
    # ------------------------------------

    plt.title("T-SNE Visualization of Feature Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # Legend settings
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Classes", borderaxespad=0.)
    plt.tight_layout()
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    # Update this path to where your features are stored
    input_dir = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/DINO_FAIR1M_features/base_classes'
    # Note: I changed the path above to 'base_classes' assuming that's where these classes (Small Car, Bus etc) live.
    # Change back to 'novel_classes' if they are in the novel folder.
    
    if os.path.exists(input_dir):
        visualize_tsne(input_dir)
    else:
        print(f"Please check the input path: {input_dir}")