import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from joblib import dump, load

def extract_hidden_features(mlp, X):
    """
    Manually pushes the data through the sklearn MLP's hidden layers 
    to extract the separated 256-D features before classification.
    """
    h1 = np.dot(X, mlp.coefs_[0]) + mlp.intercepts_[0]
    h1_relu = np.maximum(0, h1)
    
    h2 = np.dot(h1_relu, mlp.coefs_[1]) + mlp.intercepts_[1]
    h2_relu = np.maximum(0, h2)
    
    return h2_relu

def evaluate_and_visualize_superclasses(y_test, y_pred, X_kpca, X_tsne, y_test_viz):
    """
    Evaluates and visualizes the separation between 'Base' and 'Novel' classes.
    """
    # base_subclasses =[
    #     'Small Car', 'Bus', 'Dump Truck', 'Van', 'Tractor', 
    #     'Truck Tractor', 'Excavator', 'other-vehicle'
    # ]
    base_subclasses =[
        'Bus', 'Dump Truck', 'Tractor', 
        'Truck Tractor', 'Excavator', 'other-vehicle'
    ]
    
    # Helper function to map 10 classes to 2 superclasses
    def map_to_superclass(labels):
        return np.array(['Base' if lbl in base_subclasses else 'Novel' for lbl in labels])
    
    # 1. Map true labels and predicted labels
    y_test_super = map_to_superclass(y_test)
    y_pred_super = map_to_superclass(y_pred)
    
    # 2. Print Superclass Metrics
    print("\n" + "="*50)
    print("--- BASE vs NOVEL Classification Report ---")
    print(classification_report(y_test_super, y_pred_super, digits=4))
    print("="*50 + "\n")
    
    # 3. Visualization
    y_test_viz_super = map_to_superclass(y_test_viz)
    
    print("Plotting Base vs Novel separation graphs...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    unique_superclasses = ['Base', 'Novel']
    # Define color mapping explicitly
    super_palette = {'Base': 'royalblue', 'Novel': 'crimson'}
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.85)

    # --- PLOT 1: Kernel PCA (Base vs Novel) ---
    sns.scatterplot(
        ax=axes[0], x=X_kpca[:, 0], y=X_kpca[:, 1],
        hue=y_test_viz_super, palette=super_palette, s=60, alpha=0.6, edgecolor=None
    )
    for cls in unique_superclasses:
        cls_points = X_kpca[y_test_viz_super == cls]
        center_x, center_y = np.mean(cls_points[:, 0]), np.mean(cls_points[:, 1])
        # Use the exact color from the palette for the center star
        axes[0].scatter(center_x, center_y, marker='*', s=800, color=super_palette[cls], edgecolor='black', zorder=10)
        axes[0].annotate(f"{cls} Center", (center_x, center_y), xytext=(10, 10), textcoords='offset points',
                         fontsize=12, fontweight='bold', color='black', bbox=bbox_props, zorder=11)

    axes[0].set_title('Kernel PCA: Base vs Novel Classes', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(title='Superclass', bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- PLOT 2: t-SNE (Base vs Novel) ---
    sns.scatterplot(
        ax=axes[1], x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y_test_viz_super, palette=super_palette, s=60, alpha=0.6, edgecolor=None, legend=False
    )
    for cls in unique_superclasses:
        cls_points = X_tsne[y_test_viz_super == cls]
        center_x, center_y = np.median(cls_points[:, 0]), np.median(cls_points[:, 1])
        # Use the exact color from the palette for the center star
        axes[1].scatter(center_x, center_y, marker='*', s=800, color=super_palette[cls], edgecolor='black', zorder=10)
        axes[1].annotate(f"{cls} Center", (center_x, center_y), xytext=(10, 10), textcoords='offset points',
                         fontsize=12, fontweight='bold', color='black', bbox=bbox_props, zorder=11)

    axes[1].set_title('t-SNE Map: Base vs Novel Classes', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')

    plt.tight_layout()
    plt.savefig('Outputs/MLP-t-SNE-Graphs/Base_vs_Novel_Separation_NO_CARS_VANS.png', dpi=300)
    print("Done! Saved plot as 'Base_vs_Novel_Separation.png'")
    plt.show()

def main():
    # --- 1. Define Directories ---
    # Training directories (100% of these will be used for training)
    train_base_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/base_class'
    train_novel_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/novel_class'

    # Validation/Testing directories (100% of these will be used for testing)
    val_base_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/base_class'
    val_novel_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/novel_class'

    # all_classes =[
    #     'Small Car', 'Bus', 'Dump Truck', 'Van', 'Tractor', 
    #     'Truck Tractor', 'Excavator', 'other-vehicle', 
    #     'Cargo Truck', 'Trailer'
    # ]
    all_classes =[
        'Bus', 'Dump Truck', 'Tractor', 
        'Truck Tractor', 'Excavator', 
        'Cargo Truck', 'Trailer'
    ]
    safe_class_names =[cls.replace(" ", "_") for cls in all_classes]

    X_train, y_train = [], []
    X_test, y_test = [],[]

    # --- 2. Load Features ---
    def load_features_from_dir(directory, X_list, y_list):
        if not os.path.exists(directory):
            print(f"Warning: Directory does not exist -> {directory}")
            return
            
        files = glob.glob(os.path.join(directory, '*.npy'))
        for f in files:
            filename = os.path.basename(f)
            for safe_cls in safe_class_names:
                if f"_{safe_cls}_" in filename:
                    embedding = np.load(f).flatten()
                    X_list.append(embedding)
                    y_list.append(safe_cls.replace("_", " ")) 
                    break

    print("Loading 100% of Training features... (This might take a minute)")
    load_features_from_dir(train_base_dir, X_train, y_train)
    load_features_from_dir(train_novel_dir, X_train, y_train)

    print("Loading Validation/Testing features...")
    load_features_from_dir(val_base_dir, X_test, y_test)
    load_features_from_dir(val_novel_dir, X_test, y_test)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Loaded {X_train.shape[0]} training embeddings.")
    print(f"Loaded {X_test.shape[0]} validation embeddings.")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Error: Training or testing data is empty. Please check your file paths.")
        return

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # --- 3. CHECK FOR SAVED MODEL IN NEW DIRECTORY ---
    save_dir = 'models/MLP-Sklearn'
    os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist
    
    model_file = os.path.join(save_dir, 'saved_mlp_NO_CARS_VANS.joblib')
    scaler_file = os.path.join(save_dir, 'saved_scaler_NO_CARS_VANS.joblib')
    le_file = os.path.join(save_dir, 'saved_le_NO_CARS_VANS.joblib') # Save the label encoder too

    if os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(le_file):
        print(f"\nFound saved model at {save_dir}! Loading weights instantly...")
        clf = load(model_file)
        scaler = load(scaler_file)
        le = load(le_file)
        X_test_scaled = scaler.transform(X_test)
        y_test_encoded = le.transform(y_test)
    else:
        print("\nNo saved model found. Training MLP Classifier on 100% of Train Data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 

        # Using early stopping and batch_size for much faster training
        clf = MLPClassifier(
            hidden_layer_sizes=(512, 256), 
            max_iter=500,             # Keep this high, early stopping will cut it off when appropriate
            batch_size=2048,          # Processes data faster
            tol=1e-6,                 # <--- NEW: Reduced tolerance from 0.0001 to 0.000001
            n_iter_no_change=50,
            early_stopping=True,      # YES, keep this!
            validation_fraction=0.1,  # Uses 10% of train data to know when to stop
            alpha=0.01,               # NEW: L2 Regularization (forces the model to generalize, not memorize)
            random_state=42, 
            verbose=True
        )
        
        # Train on the ENCODED integers instead of strings to prevent the NaN error
        clf.fit(X_train_scaled, y_train_encoded)
        
        # Save to specific directory
        dump(clf, model_file)
        dump(scaler, scaler_file)
        dump(le, le_file)
        print(f"Training Complete! Saved model to '{model_file}'.")

    # --- 4. DETAILED METRICS CALCULATION (Tested ONLY on Validation Set) ---
    print(f"\nOverall Validation Set Accuracy: {clf.score(X_test_scaled, y_test_encoded) * 100:.2f}%")
    
    # Predict numbers, then convert back to strings for the report
    y_pred_encoded = clf.predict(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_encoded) 
    
    print("\n--- Detailed Classification Report (10 Classes) ---")
    print(classification_report(y_test, y_pred, digits=4))

    # --- 5. Sample a subset FROM THE VALIDATION SET for Visualization ---
    print("\nSampling Validation data for clear visualization...")
    X_test_viz = []
    y_test_viz =[]
    samples_per_class = 200 
    
    for cls in np.unique(y_test):
        idx = np.where(y_test == cls)[0]
        selected_idx = np.random.choice(idx, min(samples_per_class, len(idx)), replace=False)
        X_test_viz.extend(X_test_scaled[selected_idx])
        y_test_viz.extend(y_test[selected_idx])
        
    X_test_viz = np.array(X_test_viz)
    y_test_viz = np.array(y_test_viz)

    # --- 6. Extract the Separated Hidden Features ---
    print("Extracting the 256-D separated features...")
    X_separated_features = extract_hidden_features(clf, X_test_viz)

    # --- 7. Dimensionality Reduction ---
    print("Applying Kernel PCA (RBF)...")
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=None) 
    X_kpca = kpca.fit_transform(X_separated_features)

    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_separated_features)

    # --- 8. Plotting and Marking the Cluster Centers ---
    print("Plotting graphs and calculating Prototype centers...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    unique_classes = np.unique(y_test_viz)

    # Create a dictionary linking each class to a specific color from the tab10 palette
    class_palette = dict(zip(unique_classes, sns.color_palette("tab10", len(unique_classes))))
    
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.85)

    # --- PLOT 1: Kernel PCA Map ---
    sns.scatterplot(
        ax=axes[0], x=X_kpca[:, 0], y=X_kpca[:, 1],
        hue=y_test_viz, palette=class_palette, s=60, alpha=0.6, edgecolor=None
    )
    for cls in unique_classes:
        cls_points = X_kpca[y_test_viz == cls]
        center_x, center_y = np.mean(cls_points[:, 0]), np.mean(cls_points[:, 1])
        # Use exact class color for the center star
        axes[0].scatter(center_x, center_y, marker='*', s=800, color=class_palette[cls], edgecolor='black', zorder=10)
        axes[0].annotate(cls, (center_x, center_y), xytext=(8, 8), textcoords='offset points',
                         fontsize=10, fontweight='bold', color='black', bbox=bbox_props, zorder=11)

    axes[0].set_title('Kernel PCA (RBF) with Cluster Labels', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(title='Vehicle Classes', bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- PLOT 2: t-SNE Map ---
    sns.scatterplot(
        ax=axes[1], x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y_test_viz, palette=class_palette, s=60, alpha=0.6, edgecolor=None, legend=False
    )
    for cls in unique_classes:
        cls_points = X_tsne[y_test_viz == cls]
        center_x, center_y = np.median(cls_points[:, 0]), np.median(cls_points[:, 1])
        # Use exact class color for the center star
        axes[1].scatter(center_x, center_y, marker='*', s=800, color=class_palette[cls], edgecolor='black', zorder=10)
        axes[1].annotate(cls, (center_x, center_y), xytext=(8, 8), textcoords='offset points',
                         fontsize=10, fontweight='bold', color='black', bbox=bbox_props, zorder=11)

    axes[1].set_title('t-SNE Map with Cluster Labels (Prototypes)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')

    plt.tight_layout()
    plt.savefig('Outputs/MLP-t-SNE-Graphs/Test_Set_Labeled_Centers_NO_CARS_VANS.png', dpi=300)
    print("Done! Saved plot as 'Test_Set_Labeled_Centers.png'")
    plt.show()

    # === 9. Base vs Novel Evaluation and Visualization ===
    evaluate_and_visualize_superclasses(y_test, y_pred, X_kpca, X_tsne, y_test_viz)

if __name__ == "__main__":
    main()