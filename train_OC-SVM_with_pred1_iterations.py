#!/usr/bin/env python3
import numpy as np
import pickle
import argparse
import pandas as pd
import random
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION - Defaults
# ============================================================================
CLASS_0_DIR = '/work/Sultan/data/feature_vec_0/individual_features/'
CLASS_1_DIR = '/work/Sultan/data/feature_vec_1/individual_features_all/'
OUTPUT_MODEL_PATH = '/work/Sultan/models/ocsvm_model.pkl'
OUTPUT_SCALER_PATH = '/work/Sultan/models/ocsvm_scaler.pkl'
TEST_OUTPUT_CSV = '/work/Sultan/oc_svm_res/iterative_predictions_with_metrics.csv'

NUM_ITERATIONS = 20 
NORMALIZE_FEATURES = True
# ============================================================================

class OneClassSVMPredictor:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.scaler = None
        self.model = None

    def load_model_assets(self, model_path: str, scaler_path: str):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        if self.normalize:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

    def run_iterative_prediction(self, class0_path: str, class1_path: str, num_iter: int, output_csv: str):
        p0 = Path(class0_path)
        p1 = Path(class1_path)
        all_results = []
        
        # Lists to store metrics for final averaging
        all_precision = []
        all_recall = []
        all_accuracy = []

        for i in range(num_iter):
            # Randomly sample 4 from Class 0 and 2 from Class 1
            files0 = random.sample(list(p0.glob('*.npy')), 4)
            files1 = random.sample(list(p1.glob('*.npy')), 2)
            
            iteration_data = [(f, "class_0") for f in files0] + [(f, "class_1") for f in files1]
            
            tp, fp, fn, tn = 0, 0, 0, 0

            for file_path, label in iteration_data:
                feat = np.load(file_path).reshape(1, -1)
                if self.normalize and self.scaler:
                    feat = self.scaler.transform(feat)
                
                # OCSVM: 1 = Inlier (Normal), -1 = Outlier (Anomaly/Class 1)
                prediction = self.model.predict(feat)[0]
                is_outlier = (prediction == -1)
                
                # Logic Mapping: Class 1 is our "Positive" (Anomaly) class
                if label == "class_1" and is_outlier: tp += 1
                elif label == "class_0" and is_outlier: fp += 1
                elif label == "class_1" and not is_outlier: fn += 1
                elif label == "class_0" and not is_outlier: tn += 1

                all_results.append({
                    'iteration': i + 1,
                    'filename': file_path.name,
                    'source_class': label,
                    'prediction': "Outlier" if is_outlier else "Inlier",
                    'precision': "",
                    'recall': ""
                })
            
            # Calculate Iteration Metrics
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            acc = (tp + tn) / len(iteration_data)
            
            all_precision.append(prec)
            all_recall.append(rec)
            all_accuracy.append(acc)
            
            # Add Summary Row to CSV
            all_results.append({
                'iteration': f"ITER {i+1} SUMMARY",
                'filename': "---",
                'source_class': "---",
                'prediction': "---",
                'precision': f"{prec:.4f}",
                'recall': f"{rec:.4f}"
            })
            all_results.append({col: "" for col in ['iteration', 'filename', 'source_class', 'prediction', 'precision', 'recall']})

        # Final terminal output
        avg_precision = np.mean(all_precision)
        avg_recall = np.mean(all_recall)
        avg_accuracy = np.mean(all_accuracy)

        print("\n" + "="*40)
        print(f"OVERALL PERFORMANCE ({num_iter} Iterations)")
        print("="*40)
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall:    {avg_recall:.4f}")
        print(f"Average Accuracy:  {avg_accuracy:.4f}")
        print("="*40)

        # Save to CSV
        df = pd.DataFrame(all_results)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"✓ Detailed CSV saved to: {output_csv}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', type=int, default=NUM_ITERATIONS)
    parser.add_argument('--model_path', type=str, default=OUTPUT_MODEL_PATH)
    parser.add_argument('--scaler_path', type=str, default=OUTPUT_SCALER_PATH)
    parser.add_argument('--output_csv', type=str, default=TEST_OUTPUT_CSV)
    args = parser.parse_args()

    predictor = OneClassSVMPredictor(normalize=NORMALIZE_FEATURES)
    
    try:
        predictor.load_model_assets(args.model_path, args.scaler_path)
        predictor.run_iterative_prediction(CLASS_0_DIR, CLASS_1_DIR, args.num_iter, args.output_csv)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()