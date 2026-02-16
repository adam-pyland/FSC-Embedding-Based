""" I have 2 folders with feature vectors, each for a class (class0 and class1).
I would like to try to cluster them via tsne and see if they are separable. That is it, nothing more
class 0 is ~10 times more than class 1, so I will randomly sample from class 0 to have the same number of samples as class 1."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

# Paths to feature vector directories
class0_dir = '/work/Sultan/data/feature_vec_0/individual_features/'
class1_dir = '/work/Sultan/data/feature_vec_1/individual_features_all/'   

# Load feature vectors
def load_features_from_dir(directory):
    features = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            feature_path = os.path.join(directory, filename)
            feature_vector = np.load(feature_path)
            features.append(feature_vector)
    return np.array(features)       

class0_features = load_features_from_dir(class0_dir)
class1_features = load_features_from_dir(class1_dir)    

# Balance the classes by random sampling from class 0
num_class1_samples = len(class1_features)
if len(class0_features) > num_class1_samples:
    class0_features = random.sample(list(class0_features), num_class1_samples)  
    
# Combine features and create labels
X = np.vstack((class0_features, class1_features))
y = np.array([0] * len(class0_features) + [1] * len(class1_features))   
# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)  
# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[y == 0, 0], X_tsne[y == 0, 1], label='Vehicles', alpha=0.5)
plt.scatter(X_tsne[y == 1, 0], X_tsne[y == 1, 1], label='Pickup_trucks', alpha=0.5)
plt.title('t-SNE Visualization of Feature Vectors')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.show()

