#!/usr/bin/env python3
"""
One-Class SVM Training Script with Test Set Prediction

Trains a One-Class SVM (OC-SVM) on feature vectors and optionally predicts on test set.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import List, Optional
import argparse
import pandas as pd


# ============================================================================
# CONFIGURATION - Set your parameters here
# ============================================================================

# Input/Output paths
INPUT_FEATURES_DIR = '/work/Sultan/data/feature_vec_0/individual_features/'  # Directory with .npy feature files
OUTPUT_MODEL_PATH = '/work/Sultan/models/ocsvm_model.pkl'  # Where to save the trained model
OUTPUT_SCALER_PATH = '/work/Sultan/models/ocsvm_scaler.pkl'  # Where to save the scaler

# One-Class SVM parameters
NU = 0.1  # Expected proportion of outliers (0.01 = 1%, 0.1 = 10%, 0.5 = 50%)
KERNEL = 'rbf'  # Kernel type: 'rbf', 'linear', 'poly', 'sigmoid'
GAMMA = 'scale'  # Kernel coefficient: 'scale', 'auto', or float value

# Optional: Hyperparameter tuning
TUNE_HYPERPARAMETERS = True  # Set to True to search for best params (slower)
TUNE_NU_VALUES = [0.01, 0.05, 0.1, 0.2]  # Nu values to try if tuning
TUNE_GAMMA_VALUES = ['scale', 'auto', 0.001, 0.01, 0.1]  # Gamma values to try if tuning

# Data preprocessing
NORMALIZE_FEATURES = True  # Recommended: normalize features before training

# Test set prediction (NEW)
TEST_FEATURES_DIR = '/work/Sultan/data/feature_vec_1/individual_features_6/'  # Directory with test .npy files (leave empty to skip)
TEST_OUTPUT_CSV = '/work/Sultan/oc_svm_res/ocsvm_predictions.csv'  # Where to save predictions

# ============================================================================


class OneClassSVMTrainer:
    """Train and save a One-Class SVM model."""
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.model = None
        
    def load_features(self, features_dir: str) -> np.ndarray:
        """
        Load all feature vectors from directory.
        
        Args:
            features_dir: Directory containing .npy feature files
            
        Returns:
            Feature array of shape (n_samples, n_features)
        """
        features_path = Path(features_dir)
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features directory not found: {features_dir}")
        
        # Find all .npy files
        feature_files = sorted(features_path.glob('*_features.npy'))
        
        if not feature_files:
            raise ValueError(f"No feature files found in {features_dir}")
        
        print(f"Found {len(feature_files)} feature files")
        
        # Load all features
        features_list = []
        self.filenames = []  # Store filenames for later use
        for feat_file in feature_files:
            feat = np.load(feat_file)
            features_list.append(feat)
            self.filenames.append(feat_file.name)
        
        features = np.array(features_list)
        print(f"Loaded features shape: {features.shape}")
        
        return features
    
    def train(self, features: np.ndarray, nu: float, kernel: str, gamma: str):
        """
        Train One-Class SVM.
        
        Args:
            features: Training features (n_samples, n_features)
            nu: Expected proportion of outliers (0 < nu <= 1)
            kernel: Kernel type
            gamma: Kernel coefficient
        """
        print("\n" + "="*60)
        print("Training One-Class SVM")
        print("="*60)
        
        # Normalize if enabled
        if self.normalize:
            print("Normalizing features...")
            features = self.scaler.fit_transform(features)
            print(f"Feature stats after normalization: mean={features.mean():.4f}, std={features.std():.4f}")
        
        # Train model
        print(f"\nTraining with parameters:")
        print(f"  nu: {nu}")
        print(f"  kernel: {kernel}")
        print(f"  gamma: {gamma}")
        
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        
        self.model.fit(features)
        
        # Evaluate on training data
        predictions = self.model.predict(features)
        n_inliers = np.sum(predictions == 1)
        n_outliers = np.sum(predictions == -1)
        outlier_ratio = n_outliers / len(predictions)
        
        print(f"\nTraining Results:")
        print(f"  Total samples: {len(features)}")
        print(f"  Classified as inliers: {n_inliers} ({100*n_inliers/len(features):.1f}%)")
        print(f"  Classified as outliers: {n_outliers} ({100*outlier_ratio:.1f}%)")
        print(f"  Number of support vectors: {len(self.model.support_vectors_)}")
        
    def tune_hyperparameters(self, features: np.ndarray, 
                            nu_values: List[float], 
                            gamma_values: List):
        """
        Tune hyperparameters using grid search.
        
        Args:
            features: Training features
            nu_values: List of nu values to try
            gamma_values: List of gamma values to try
        """
        print("\n" + "="*60)
        print("Tuning Hyperparameters")
        print("="*60)
        
        if self.normalize:
            features = self.scaler.fit_transform(features)
        
        param_grid = {
            'nu': nu_values,
            'gamma': gamma_values
        }
        
        print(f"Testing {len(nu_values)} nu values × {len(gamma_values)} gamma values")
        print(f"Total combinations: {len(nu_values) * len(gamma_values)}")
        
        grid_search = GridSearchCV(
            OneClassSVM(kernel=KERNEL),
            param_grid,
            cv=3,
            scoring='accuracy',
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(features)
        
        print(f"\nBest parameters found:")
        print(f"  nu: {grid_search.best_params_['nu']}")
        print(f"  gamma: {grid_search.best_params_['gamma']}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def predict_test_set(self, test_dir: str, output_csv: str):
        """
        Predict on test set and save results to CSV.
        
        Args:
            test_dir: Directory containing test .npy files
            output_csv: Path to save predictions CSV
        """
        print("\n" + "="*60)
        print("Predicting on Test Set")
        print("="*60)
        
        # Load test features
        test_features_path = Path(test_dir)
        test_files = sorted(test_features_path.glob('*_features.npy'))
        
        if not test_files:
            print(f"No test files found in {test_dir}")
            return
        
        print(f"Found {len(test_files)} test files")
        
        # Load test features
        test_features_list = []
        test_filenames = []
        for test_file in test_files:
            feat = np.load(test_file)
            test_features_list.append(feat)
            test_filenames.append(test_file.name)
        
        test_features = np.array(test_features_list)
        print(f"Test features shape: {test_features.shape}")
        
        # Normalize if needed
        if self.normalize and self.scaler is not None:
            test_features = self.scaler.transform(test_features)
        
        # Predict
        predictions = self.model.predict(test_features)
        scores = self.model.decision_function(test_features)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'filename': test_filenames,
            'prediction': predictions,
            'is_inlier': predictions == 1,
            'is_outlier': predictions == -1,
            'decision_score': scores
        })
        
        # Save to CSV
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        
        # Print summary
        n_inliers = np.sum(predictions == 1)
        n_outliers = np.sum(predictions == -1)
        
        print(f"\nTest Set Results:")
        print(f"  Total test samples: {len(test_features)}")
        print(f"  Predicted as inliers: {n_inliers} ({100*n_inliers/len(test_features):.1f}%)")
        print(f"  Predicted as outliers: {n_outliers} ({100*n_outliers/len(test_features):.1f}%)")
        print(f"✓ Predictions saved to: {output_csv}")
    
    def save_model(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Save trained model and scaler to disk.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler (if normalization is used)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create output directory if needed
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save scaler if normalization is used
        if self.normalize and scaler_path:
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✓ Scaler saved to: {scaler_path}")


def main():
    parser = argparse.ArgumentParser(description='Train One-Class SVM on feature vectors')
    
    # All arguments have defaults from the configuration section
    parser.add_argument('--input_dir', type=str, default=INPUT_FEATURES_DIR,
                       help='Directory with feature .npy files')
    parser.add_argument('--output_model', type=str, default=OUTPUT_MODEL_PATH,
                       help='Output path for trained model')
    parser.add_argument('--output_scaler', type=str, default=OUTPUT_SCALER_PATH,
                       help='Output path for scaler')
    parser.add_argument('--nu', type=float, default=NU,
                       help='Expected proportion of outliers (0 < nu <= 1)')
    parser.add_argument('--kernel', type=str, default=KERNEL,
                       choices=['rbf', 'linear', 'poly', 'sigmoid'],
                       help='Kernel type')
    parser.add_argument('--gamma', type=str, default=GAMMA,
                       help='Kernel coefficient (scale, auto, or float)')
    parser.add_argument('--normalize', type=bool, default=NORMALIZE_FEATURES,
                       help='Normalize features before training')
    parser.add_argument('--tune', type=bool, default=TUNE_HYPERPARAMETERS,
                       help='Tune hyperparameters with grid search')
    
    # NEW: Test set arguments
    parser.add_argument('--test_dir', type=str, default=TEST_FEATURES_DIR,
                       help='Directory with test feature .npy files')
    parser.add_argument('--test_output', type=str, default=TEST_OUTPUT_CSV,
                       help='Output CSV file for test predictions')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ONE-CLASS SVM TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Input features: {args.input_dir}")
    print(f"  Output model: {args.output_model}")
    print(f"  Output scaler: {args.output_scaler}")
    print(f"  Normalize: {args.normalize}")
    print(f"  Tune hyperparameters: {args.tune}")
    if args.test_dir:
        print(f"  Test directory: {args.test_dir}")
        print(f"  Test output: {args.test_output}")
    
    # Initialize trainer
    trainer = OneClassSVMTrainer(normalize=args.normalize)
    
    # Load features
    print("\n" + "="*60)
    print("Loading Features")
    print("="*60)
    features = trainer.load_features(args.input_dir)
    
    # Train model
    if args.tune:
        # Hyperparameter tuning
        best_params = trainer.tune_hyperparameters(
            features,
            nu_values=TUNE_NU_VALUES,
            gamma_values=TUNE_GAMMA_VALUES
        )
    else:
        # Train with specified parameters
        trainer.train(features, nu=args.nu, kernel=args.kernel, gamma=args.gamma)
    
    # Save model
    trainer.save_model(args.output_model, args.output_scaler if args.normalize else None)
    
    # NEW: Predict on test set if provided
    if args.test_dir:
        trainer.predict_test_set(args.test_dir, args.test_output)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
