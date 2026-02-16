#!/usr/bin/env python3
"""
Compare feature vectors from two directories and against an averaged feature vector.
Results are saved to a CSV file.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cosine
import glob

def load_npy_files(directory):
    """Load all .npy files from a directory."""
    vectors = {}
    npy_files = glob.glob(os.path.join(directory, "*.npy"))
    
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        vector_name = os.path.splitext(filename)[0]
        try:
            vector = np.load(file_path)
            vectors[vector_name] = vector.flatten()  # Ensure 1D
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return vectors

def compute_similarity_metrics(vec1, vec2):
    """Compute various similarity/distance metrics between two vectors."""
    # Ensure vectors are 1D and same shape
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    # Cosine similarity
    cos_sim = cosine_similarity([vec1], [vec2])[0][0]
    
    # Euclidean distance
    euc_dist = euclidean(vec1, vec2)
    
    # Cosine distance
    cos_dist = cosine(vec1, vec2)
    
    # L1 distance (Manhattan)
    l1_dist = np.sum(np.abs(vec1 - vec2))
    
    # Dot product
    dot_product = np.dot(vec1, vec2)
    
    return {
        'cosine_similarity': cos_sim,
        'euclidean_distance': euc_dist,
        'cosine_distance': cos_dist,
        'l1_distance': l1_dist,
        'dot_product': dot_product
    }

def main():
    parser = argparse.ArgumentParser(
        description='Compare feature vectors and save results to CSV'
    )
    
    # Default paths
    default_vec1_dir = '/work/Sultan/data/feature_vec_0/individual_feature_test/'
    default_vec0_dir = '/work/Sultan/data/feature_vec_0/individual_features/'
    default_avg_file = '/work/Sultan/data/feature_vec_0/averaged_features.npy'
    default_output = '/work/Sultan/similarity/feature_comparison_results_control.csv'
    
    parser.add_argument(
        '--vec1-dir',
        type=str,
        default=default_vec1_dir,
        help=f'Directory containing feature vectors to compare (default: {default_vec1_dir})'
    )
    
    parser.add_argument(
        '--vec0-dir',
        type=str,
        default=default_vec0_dir,
        help=f'Directory containing reference feature vectors (default: {default_vec0_dir})'
    )
    
    parser.add_argument(
        '--avg-file',
        type=str,
        default=default_avg_file,
        help=f'Path to averaged features file (default: {default_avg_file})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=default_output,
        help=f'Output CSV file path (default: {default_output})'
    )
    
    parser.add_argument(
        '--include-individual',
        action='store_true',
        default=True,
        help='Include comparison with individual reference vectors (default: True)'
    )
    
    parser.add_argument(
        '--include-averaged',
        action='store_true',
        default=True,
        help='Include comparison with averaged reference vector (default: True)'
    )
    
    args = parser.parse_args()
    
    print("Loading feature vectors...")
    
    # Load vectors from vec1 directory
    print(f"Loading vectors from: {args.vec1_dir}")
    vec1_vectors = load_npy_files(args.vec1_dir)
    print(f"Loaded {len(vec1_vectors)} vectors from vec1 directory")
    
    # Load vectors from vec0 directory
    print(f"Loading vectors from: {args.vec0_dir}")
    vec0_vectors = load_npy_files(args.vec0_dir)
    print(f"Loaded {len(vec0_vectors)} vectors from vec0 directory")
    
    # Load averaged vector
    print(f"Loading averaged vector from: {args.avg_file}")
    try:
        averaged_vector = np.load(args.avg_file).flatten()
        print("Loaded averaged vector successfully")
    except Exception as e:
        print(f"Error loading averaged vector: {e}")
        averaged_vector = None
        if args.include_averaged:
            print("Warning: Disabling averaged comparison due to loading error")
            args.include_averaged = False
    
    # Prepare results list
    results = []
    
    # Compare each vector in vec1 with each vector in vec0
    if args.include_individual:
        print(f"\nComparing {len(vec1_vectors)} vectors with {len(vec0_vectors)} reference vectors...")
        for vec1_name, vec1_data in vec1_vectors.items():
            for vec0_name, vec0_data in vec0_vectors.items():
                try:
                    metrics = compute_similarity_metrics(vec1_data, vec0_data)
                    result = {
                        'source_vector': vec1_name,
                        'reference_type': 'individual',
                        'reference_name': vec0_name,
                        **metrics
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error comparing {vec1_name} with {vec0_name}: {e}")
    
    # Compare each vector in vec1 with averaged vector
    if args.include_averaged and averaged_vector is not None:
        print(f"\nComparing {len(vec1_vectors)} vectors with averaged vector...")
        for vec1_name, vec1_data in vec1_vectors.items():
            try:
                metrics = compute_similarity_metrics(vec1_data, averaged_vector)
                result = {
                    'source_vector': vec1_name,
                    'reference_type': 'averaged',
                    'reference_name': 'averaged_features',
                    **metrics
                }
                results.append(result)
            except Exception as e:
                print(f"Error comparing {vec1_name} with averaged vector: {e}")
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        column_order = [
            'source_vector', 
            'reference_type', 
            'reference_name',
            'cosine_similarity',
            'cosine_distance',
            'euclidean_distance',
            'l1_distance',
            'dot_product'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        # Sort by source_vector and reference_type
        df = df.sort_values(['source_vector', 'reference_type', 'reference_name'])
        
        # Save to CSV
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
        print(f"Total comparisons: {len(df)}")
        print(f"Columns saved: {', '.join(df.columns)}")
        
        # Display summary statistics
        print("\nSummary Statistics:")
        if 'cosine_similarity' in df.columns:
            print(f"Cosine Similarity - Mean: {df['cosine_similarity'].mean():.4f}, "
                  f"Std: {df['cosine_similarity'].std():.4f}")
        if 'euclidean_distance' in df.columns:
            print(f"Euclidean Distance - Mean: {df['euclidean_distance'].mean():.4f}, "
                  f"Std: {df['euclidean_distance'].std():.4f}")
    else:
        print("\nNo results generated. Check if vectors were loaded successfully.")
    
    print("\nDone!")

if __name__ == "__main__":
    main()