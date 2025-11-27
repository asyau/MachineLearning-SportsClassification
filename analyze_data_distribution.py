"""
Sports Classification Dataset - Data Distribution Analysis
This script analyzes the distribution of images across classes and splits,
identifies imbalances, and generates visualizations and CSV reports.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def count_images_in_directory(directory_path):
    """
    Count images in each subdirectory (class) of the given directory.
    
    Args:
        directory_path: Path to the directory containing class subdirectories
        
    Returns:
        Dictionary with class names as keys and image counts as values
    """
    class_counts = {}
    
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist!")
        return class_counts
    
    # Iterate through all subdirectories (each represents a class)
    for class_name in sorted(os.listdir(directory_path)):
        class_path = os.path.join(directory_path, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        # Count image files in the class directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        image_count = sum(1 for file in os.listdir(class_path) 
                         if os.path.splitext(file.lower())[1] in image_extensions)
        
        class_counts[class_name] = image_count
    
    return class_counts

def analyze_dataset(base_path):
    """
    Analyze the entire dataset across train, validation, and test splits.
    
    Args:
        base_path: Root directory containing train/valid/test folders
        
    Returns:
        DataFrame with complete distribution information
    """
    # Define split directories
    splits = ['train', 'valid', 'test']
    
    # Collect data for all splits
    data = defaultdict(dict)
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        counts = count_images_in_directory(split_path)
        
        for class_name, count in counts.items():
            data[class_name][split] = count
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.fillna(0).astype(int)
    
    # Ensure all splits are present
    for split in splits:
        if split not in df.columns:
            df[split] = 0
    
    # Calculate total and statistics
    df['total'] = df.sum(axis=1)
    df['train_percentage'] = (df['train'] / df['total'] * 100).round(2)
    df['valid_percentage'] = (df['valid'] / df['total'] * 100).round(2)
    df['test_percentage'] = (df['test'] / df['total'] * 100).round(2)
    
    # Sort by total count descending
    df = df.sort_values('total', ascending=False)
    
    # Reset index to make class name a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'class_name'}, inplace=True)
    
    return df

def calculate_imbalance_metrics(df):
    """
    Calculate various imbalance metrics for the dataset.
    
    Args:
        df: DataFrame with distribution data
        
    Returns:
        Dictionary containing imbalance metrics
    """
    metrics = {}
    
    for split in ['train', 'valid', 'test', 'total']:
        if split not in df.columns:
            continue
            
        values = df[split]
        max_count = values.max()
        min_count = values[values > 0].min() if (values > 0).any() else 0
        mean_count = values.mean()
        median_count = values.median()
        std_count = values.std()
        
        # Imbalance ratio
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Coefficient of variation
        cv = (std_count / mean_count * 100) if mean_count > 0 else 0
        
        metrics[split] = {
            'max': max_count,
            'min': min_count,
            'mean': mean_count,
            'median': median_count,
            'std': std_count,
            'imbalance_ratio': imbalance_ratio,
            'coefficient_of_variation': cv
        }
    
    return metrics

def plot_distribution_by_split(df, output_dir='output'):
    """
    Create bar plots showing distribution for each split.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot for each split
    splits = ['train', 'valid', 'test']
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 20))
    
    for idx, split in enumerate(splits):
        if split not in df.columns:
            continue
        
        # Sort by count for this split
        df_sorted = df.sort_values(split, ascending=False)
        
        ax = axes[idx]
        bars = ax.bar(range(len(df_sorted)), df_sorted[split], 
                      color=sns.color_palette("husl", 1)[0], alpha=0.8)
        
        # Highlight top 5 and bottom 5
        for i in range(min(5, len(bars))):
            bars[i].set_color('darkgreen')
            bars[-(i+1)].set_color('darkred')
        
        ax.set_xlabel('Class Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title(f'{split.capitalize()} Set Distribution (Total: {df_sorted[split].sum()} images)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean and median lines
        mean_val = df_sorted[split].mean()
        median_val = df_sorted[split].median()
        ax.axhline(y=mean_val, color='blue', linestyle='--', label=f'Mean: {mean_val:.1f}', linewidth=2)
        ax.axhline(y=median_val, color='orange', linestyle='--', label=f'Median: {median_val:.1f}', linewidth=2)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_by_split.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/distribution_by_split.png")
    plt.close()

def plot_top_bottom_classes(df, output_dir='output'):
    """
    Create horizontal bar plots showing top and bottom classes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    
    # Top 20 classes
    top_20 = df.nlargest(20, 'total')
    ax1 = axes[0]
    bars = ax1.barh(range(len(top_20)), top_20['total'], color='green', alpha=0.7)
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20['class_name'], fontsize=10)
    ax1.set_xlabel('Total Images', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Classes by Image Count', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax1.text(row['total'] + 2, i, str(row['total']), 
                va='center', fontsize=9, fontweight='bold')
    
    # Bottom 20 classes
    bottom_20 = df.nsmallest(20, 'total')
    ax2 = axes[1]
    bars = ax2.barh(range(len(bottom_20)), bottom_20['total'], color='red', alpha=0.7)
    ax2.set_yticks(range(len(bottom_20)))
    ax2.set_yticklabels(bottom_20['class_name'], fontsize=10)
    ax2.set_xlabel('Total Images', fontsize=12, fontweight='bold')
    ax2.set_title('Bottom 20 Classes by Image Count', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(bottom_20.iterrows()):
        ax2.text(row['total'] + 2, i, str(row['total']), 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_bottom_classes.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/top_bottom_classes.png")
    plt.close()

def plot_heatmap(df, output_dir='output'):
    """
    Create a heatmap showing the distribution across classes and splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for heatmap
    heatmap_data = df[['class_name', 'train', 'valid', 'test']].set_index('class_name')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 24))
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Number of Images'})
    
    ax.set_title('Image Distribution Heatmap (All Classes)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Data Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class Name', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/distribution_heatmap.png")
    plt.close()

def plot_split_ratios(df, output_dir='output'):
    """
    Create pie charts showing the overall split ratios.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate total images per split
    split_totals = {
        'Train': df['train'].sum(),
        'Validation': df['valid'].sum(),
        'Test': df['test'].sum()
    }
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(split_totals.values(), 
                                       labels=split_totals.keys(),
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       explode=explode,
                                       startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Add count in the label
    for i, (label, count) in enumerate(split_totals.items()):
        texts[i].set_text(f'{label}\n({count} images)')
    
    ax.set_title('Dataset Split Distribution', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_ratio.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/split_ratio.png")
    plt.close()

def plot_imbalance_analysis(df, output_dir='output'):
    """
    Create visualization showing class imbalance.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution histogram for total images
    ax1 = axes[0, 0]
    ax1.hist(df['total'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df['total'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['total'].mean():.1f}")
    ax1.axvline(df['total'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {df['total'].median():.1f}")
    ax1.set_xlabel('Number of Images per Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Classes', fontsize=11, fontweight='bold')
    ax1.set_title('Total Images Distribution Across Classes', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Box plot for each split
    ax2 = axes[0, 1]
    box_data = [df['train'], df['valid'], df['test']]
    bp = ax2.boxplot(box_data, labels=['Train', 'Valid', 'Test'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightcoral', 'lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution Box Plot by Split', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_totals = np.sort(df['total'])
    cumulative = np.arange(1, len(sorted_totals) + 1) / len(sorted_totals) * 100
    ax3.plot(sorted_totals, cumulative, linewidth=2, color='purple')
    ax3.set_xlabel('Number of Images', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cumulative Percentage of Classes', fontsize=11, fontweight='bold')
    ax3.set_title('Cumulative Distribution of Images per Class', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Train/Valid/Test ratio per class (scatter)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['train_percentage'], df['test_percentage'], 
                         c=df['valid_percentage'], cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='black')
    ax4.set_xlabel('Train Percentage (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Test Percentage (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Split Ratios per Class (color = Valid %)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Valid %', fontsize=10, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'imbalance_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/imbalance_analysis.png")
    plt.close()

def generate_summary_report(df, metrics, output_dir='output'):
    """
    Generate a text summary report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPORTS CLASSIFICATION DATASET - DISTRIBUTION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total number of classes: {len(df)}\n")
        f.write(f"Total images: {df['total'].sum()}\n")
        f.write(f"  - Train: {df['train'].sum()} images\n")
        f.write(f"  - Valid: {df['valid'].sum()} images\n")
        f.write(f"  - Test: {df['test'].sum()} images\n\n")
        
        # Split ratios
        total_images = df['total'].sum()
        f.write("SPLIT RATIOS\n")
        f.write("-"*80 + "\n")
        f.write(f"Train: {df['train'].sum()/total_images*100:.2f}%\n")
        f.write(f"Valid: {df['valid'].sum()/total_images*100:.2f}%\n")
        f.write(f"Test: {df['test'].sum()/total_images*100:.2f}%\n\n")
        
        # Imbalance metrics
        f.write("IMBALANCE METRICS\n")
        f.write("-"*80 + "\n")
        for split, split_metrics in metrics.items():
            f.write(f"\n{split.upper()}:\n")
            f.write(f"  Max images per class: {split_metrics['max']}\n")
            f.write(f"  Min images per class: {split_metrics['min']}\n")
            f.write(f"  Mean images per class: {split_metrics['mean']:.2f}\n")
            f.write(f"  Median images per class: {split_metrics['median']:.2f}\n")
            f.write(f"  Standard deviation: {split_metrics['std']:.2f}\n")
            f.write(f"  Imbalance ratio (max/min): {split_metrics['imbalance_ratio']:.2f}\n")
            f.write(f"  Coefficient of variation: {split_metrics['coefficient_of_variation']:.2f}%\n")
        
        # Most and least represented classes
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 MOST REPRESENTED CLASSES\n")
        f.write("-"*80 + "\n")
        top_10 = df.nlargest(10, 'total')
        for idx, row in top_10.iterrows():
            f.write(f"{row['class_name']:30s} : {row['total']:4d} images "
                   f"(Train: {row['train']}, Valid: {row['valid']}, Test: {row['test']})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TOP 10 LEAST REPRESENTED CLASSES\n")
        f.write("-"*80 + "\n")
        bottom_10 = df.nsmallest(10, 'total')
        for idx, row in bottom_10.iterrows():
            f.write(f"{row['class_name']:30s} : {row['total']:4d} images "
                   f"(Train: {row['train']}, Valid: {row['valid']}, Test: {row['test']})\n")
        
        # Recommendations
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        
        imbalance_ratio = metrics['total']['imbalance_ratio']
        if imbalance_ratio > 3:
            f.write("⚠ HIGH CLASS IMBALANCE DETECTED!\n")
            f.write(f"  The imbalance ratio is {imbalance_ratio:.2f}, which indicates significant disparity.\n")
            f.write("  Consider:\n")
            f.write("  1. Using class weights during training\n")
            f.write("  2. Applying data augmentation to underrepresented classes\n")
            f.write("  3. Using techniques like oversampling (SMOTE) or undersampling\n")
            f.write("  4. Using focal loss or other imbalance-aware loss functions\n")
        elif imbalance_ratio > 1.5:
            f.write("⚠ MODERATE CLASS IMBALANCE DETECTED\n")
            f.write(f"  The imbalance ratio is {imbalance_ratio:.2f}.\n")
            f.write("  Consider using class weights or data augmentation.\n")
        else:
            f.write("✓ Dataset appears relatively balanced.\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved: {report_path}")

def main():
    """
    Main function to run the complete analysis.
    """
    # Get the base path (assumes script is in the root directory)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    print("="*80)
    print("SPORTS CLASSIFICATION DATASET - DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing dataset at: {base_path}\n")
    
    # Analyze dataset
    print("📊 Analyzing dataset structure...")
    df = analyze_dataset(base_path)
    
    print(f"✓ Found {len(df)} classes")
    print(f"✓ Total images: {df['total'].sum()}")
    
    # Calculate imbalance metrics
    print("\nCalculating imbalance metrics...")
    metrics = calculate_imbalance_metrics(df)
    
    # Save to CSV
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'data_distribution.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved distribution data to: {csv_path}")
    
    # Create detailed CSV with additional statistics
    detailed_csv_path = os.path.join(output_dir, 'detailed_distribution.csv')
    df_detailed = df.copy()
    
    # Add imbalance indicators
    mean_total = df['total'].mean()
    df_detailed['above_mean'] = df_detailed['total'] > mean_total
    df_detailed['deviation_from_mean'] = df_detailed['total'] - mean_total
    df_detailed['deviation_percentage'] = (df_detailed['deviation_from_mean'] / mean_total * 100).round(2)
    
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"✓ Saved detailed distribution data to: {detailed_csv_path}")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    plot_distribution_by_split(df, output_dir)
    plot_top_bottom_classes(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_split_ratios(df, output_dir)
    plot_imbalance_analysis(df, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, metrics, output_dir)
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Classes: {len(df)}")
    print(f"Total Images: {df['total'].sum()}")
    print(f"  - Train: {df['train'].sum()} ({df['train'].sum()/df['total'].sum()*100:.1f}%)")
    print(f"  - Valid: {df['valid'].sum()} ({df['valid'].sum()/df['total'].sum()*100:.1f}%)")
    print(f"  - Test: {df['test'].sum()} ({df['test'].sum()/df['total'].sum()*100:.1f}%)")
    print(f"\nImbalance Ratio: {metrics['total']['imbalance_ratio']:.2f}")
    print(f"Mean images per class: {metrics['total']['mean']:.2f}")
    print(f"Std deviation: {metrics['total']['std']:.2f}")
    
    print("\n" + "="*80)
    print(f"✓ Analysis complete! All outputs saved to '{output_dir}/' directory")
    print("="*80)

if __name__ == "__main__":
    main()

