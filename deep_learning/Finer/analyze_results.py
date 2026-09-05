#!/usr/bin/env python3
"""
SCGINR vs SPINR Results Analysis Script
Analyzes and compares experimental results from SCGINR and SPINR models
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_dir='results/single_exp'):
    """Load all experiment results from JSON files"""
    results = []

    # Find all result JSON files
    json_files = glob.glob(os.path.join(results_dir, '**/PaviaU_*_results_Q8.json'), recursive=True)

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract key information
            args = data.get('args', {})
            model_type = args.get('model_type', 'Unknown')
            pool_size = args.get('pool_size', 'N/A')
            learning_rate = args.get('learning_rate', 'N/A')

            # Get quantized results
            quant_results = data['results'].get('quantized', {})
            if quant_results:
                result_entry = {
                    'model_type': model_type,
                    'pool_size': pool_size,
                    'learning_rate': learning_rate,
                    'psnr': quant_results.get('psnr', 0),
                    'ssim': quant_results.get('ssim', 0),
                    'sam': quant_results.get('sam', 0),
                    'final_bpppb': quant_results.get('final_bpppb', 0),
                    'compression_ratio': quant_results.get('compression_ratio', 0),
                    'encode_time': quant_results.get('encode_time', 0),
                    'decode_time': quant_results.get('decode_time', 0),
                    'json_file': json_file
                }
                results.append(result_entry)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results

def analyze_results(results):
    """Analyze and display results"""
    if not results:
        print("No results found!")
        return

    df = pd.DataFrame(results)

    print("=" * 80)
    print("SCGINR vs SPINR Experimental Results Summary")
    print("=" * 80)
    print()

    # Separate SCGINR and SPINR results
    scginr_results = df[df['model_type'] == 'SCGINR']
    spinr_results = df[df['model_type'] == 'SPINR']

    if not spinr_results.empty:
        print("SPINR Baseline:")
        print("-" * 80)
        for _, row in spinr_results.iterrows():
            print(f"  PSNR: {row['psnr']:.4f} dB")
            print(f"  SSIM: {row['ssim']:.4f}")
            print(f"  SAM: {row['sam']:.4f} rad")
            print(f"  bpppb: {row['final_bpppb']:.4f}")
            print(f"  Compression Ratio: {row['compression_ratio']:.2f}x")
            print(f"  Encode Time: {row['encode_time']:.4f}s")
            print(f"  Decode Time: {row['decode_time']:.4f}s")
        print()

    if not scginr_results.empty:
        print("SCGINR Results:")
        print("-" * 80)
        for idx, row in scginr_results.iterrows():
            print(f"\nExperiment: pool_size={row['pool_size']}, lr={row['learning_rate']}")
            print(f"  PSNR: {row['psnr']:.4f} dB", end="")
            if not spinr_results.empty:
                psnr_diff = row['psnr'] - spinr_results.iloc[0]['psnr']
                print(f" (Δ{psnr_diff:+.4f} dB)")
            else:
                print()
            print(f"  SSIM: {row['ssim']:.4f}")
            print(f"  SAM: {row['sam']:.4f} rad")
            print(f"  bpppb: {row['final_bpppb']:.4f}")
            print(f"  Compression Ratio: {row['compression_ratio']:.2f}x")
            print(f"  Encode Time: {row['encode_time']:.4f}s")
            print(f"  Decode Time: {row['decode_time']:.4f}s")
        print()

    # Find best SCGINR result
    if not scginr_results.empty:
        best_scginr = scginr_results.loc[scginr_results['psnr'].idxmax()]
        print("=" * 80)
        print("Best SCGINR Configuration:")
        print("-" * 80)
        print(f"  pool_size: {best_scginr['pool_size']}")
        print(f"  learning_rate: {best_scginr['learning_rate']}")
        print(f"  PSNR: {best_scginr['psnr']:.4f} dB")

        if not spinr_results.empty:
            spinr_psnr = spinr_results.iloc[0]['psnr']
            psnr_improvement = best_scginr['psnr'] - spinr_psnr
            print(f"  PSNR Improvement over SPINR: {psnr_improvement:+.4f} dB")

            if psnr_improvement >= -1.0:
                print(f"  ✓ Meets requirement (within 1dB of SPINR)")
            else:
                print(f"  ✗ Does not meet requirement (>1dB drop)")
        print("=" * 80)

    return df

def plot_training_curves(results_dir='results/single_exp'):
    """Plot training curves for comparison"""
    # Find all training log files
    log_files = glob.glob(os.path.join(results_dir, '**/PaviaU_*_training_logs.json'), recursive=True)

    if not log_files:
        print("No training logs found for plotting.")
        return

    plt.figure(figsize=(12, 6))

    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)

            # Extract model info from filename
            filename = os.path.basename(log_file)
            if 'SCGINR' in filename:
                if 'pool3' in filename:
                    label = 'SCGINR (pool=3)'
                elif 'pool5' in filename:
                    label = 'SCGINR (pool=5)'
                elif 'pool7' in filename:
                    label = 'SCGINR (pool=7)'
                else:
                    label = 'SCGINR'
            elif 'SPINR' in filename:
                label = 'SPINR (baseline)'
            else:
                label = 'Unknown'

            psnr_values = logs.get('psnr', [])
            if psnr_values:
                plt.plot(psnr_values, label=label, linewidth=2, alpha=0.8)

        except Exception as e:
            print(f"Error loading {log_file}: {e}")

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('SCGINR vs SPINR Training Curves (PaviaU)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(results_dir, 'scginr_vs_spinr_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves plot saved to: {output_path}")
    plt.close()

def main():
    results_dir = 'results/single_exp'

    # Load and analyze results
    results = load_results(results_dir)
    df = analyze_results(results)

    # Plot training curves
    if df is not None and not df.empty:
        plot_training_curves(results_dir)

    # Save summary to CSV
    if df is not None and not df.empty:
        csv_path = os.path.join(results_dir, 'scginr_vs_spinr_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults summary saved to: {csv_path}")

if __name__ == '__main__':
    main()
