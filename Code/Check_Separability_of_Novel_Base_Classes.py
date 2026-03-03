import os
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def check_embedding_separability():
    base_dir = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/DINO_FAIR1M_features/base_classes'
    novel_dir = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/DINO_FAIR1M_features/novel_classes'

    all_classes = [
        'Small Car', 'Bus', 'Dump Truck', 'Van', 'Tractor', 
        'Truck Tractor', 'Excavator', 'other-vehicle', 
        'Cargo Truck', 'Trailer'
    ]
    safe_class_names = [cls.replace(" ", "_") for cls in all_classes]

    X = []
    y = []

    # Helper to load features
    def load_features_from_dir(directory):
        files = glob.glob(os.path.join(directory, '*.npy'))
        for f in files:
            filename = os.path.basename(f)
            # Find which class this file belongs to
            for safe_cls in safe_class_names:
                if f"_{safe_cls}_" in filename:
                    embedding = np.load(f)
                    # Handle shapes: if it's (1, 1024), flatten it to (1024,)
                    X.append(embedding.flatten())
                    y.append(safe_cls)
                    break

    print("Loading features...")
    load_features_from_dir(base_dir)
    load_features_from_dir(novel_dir)

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {X.shape[0]} embeddings of shape {X.shape[1]}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the data mathematically so the LR model can read it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Train a simple linear classifier
    print("Training Logistic Regression...")
    # ====================================================================
    # The Separation algorthims options
    # clf = LogisticRegression(max_iter=3000, class_weight='balanced')
    clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=500, random_state=42)
    # ====================================================================
    clf.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nOverall Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

check_embedding_separability()