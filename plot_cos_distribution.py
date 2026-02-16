#!/usr/bin/env python3
"""
Simple histogram comparing cosine similarity distributions from two CSV files.
Red: control group (common vs common)
Green: experiment group (rare vs common)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Compare cosine similarity distributions')
    parser.add_argument('--control', type=str, default='/work/Sultan/similarity/feature_comparison_results_control.csv',
                       help='CSV file with control group comparisons (common vs common)')
    parser.add_argument('--experiment', type=str, default='/work/Sultan/similarity/feature_comparison_results.csv',
                       help='CSV file with experiment group comparisons (rare vs common)')
    parser.add_argument('--output', type=str, default='similarity_comparison.png',
                       help='Output image file')
    parser.add_argument('--bins', type=int, default=30,
                       help='Number of histogram bins')
    
    args = parser.parse_args()
    
    # Load data
    df_control = pd.read_csv(args.control)
    df_exp = pd.read_csv(args.experiment)
    
    # Get cosine similarities
    control_sim = df_control['cosine_similarity'].values
    exp_sim = df_exp['cosine_similarity'].values
    
    # Print statistics
    print(f"Control (common vs common): n={len(control_sim)}, mean={control_sim.mean():.3f}, std={control_sim.std():.3f}")
    print(f"Experiment (rare vs common): n={len(exp_sim)}, mean={exp_sim.mean():.3f}, std={exp_sim.std():.3f}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    # Plot both histograms
    plt.hist(control_sim, bins=args.bins, alpha=0.7, color='red', 
             label=f'Control (common-common)', edgecolor='black', linewidth=0.5)
    plt.hist(exp_sim, bins=args.bins, alpha=0.7, color='green', 
             label=f'Experiment (rare-common)', edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for means
    plt.axvline(control_sim.mean(), color='darkred', linestyle='--', 
                label=f'Control mean: {control_sim.mean():.3f}')
    plt.axvline(exp_sim.mean(), color='darkgreen', linestyle='--', 
                label=f'Experiment mean: {exp_sim.mean():.3f}')
    
    # Labels and title
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Cosine Similarity Distribution:\nControl vs Experiment', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved histogram to {args.output}")
    plt.show()

if __name__ == "__main__":
    main()