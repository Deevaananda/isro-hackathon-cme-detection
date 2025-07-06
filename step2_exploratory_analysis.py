#!/usr/bin/env python3
"""
ISRO Hackathon PS10 - Step 2: Exploratory Data Analysis
Halo CME Detection using ADITYA-L1 SWIS-ASPEX Data

Based on requirements:
- Analyze flux, density, temperature, and speed parameters during CME windows
- Examine parameter distributions and correlations
- Identify patterns a        f.write(f"CME periods (precise): {df['cme_event'].sum():,} points ({df['cme_event'].mean()*100:.2f}%)\n")
        f.write(f"CME periods (extended): {df['cme_event_extended'].sum():,} points ({df['cme_event_extended'].mean()*100:.2f}%)\n\n") anomalies
- Generate comprehensive visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = "cme_detection_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("="*80)
print("ISRO HACKATHON PS10 - HALO CME DETECTION")
print("Step 2: Exploratory Data Analysis")
print("="*80)

def load_prepared_data():
    """Load the prepared datasets from Step 1"""
    print("Loading prepared datasets from Step 1...")
    
    try:
        # Load main dataset
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "integrated_swis_data.csv"))
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Load CME events
        cme_events = pd.read_csv(os.path.join(OUTPUT_DIR, "cactus_cme_events.csv"))
        cme_events['datetime'] = pd.to_datetime(cme_events['datetime'])
        
        # Load CME windows
        cme_windows = pd.read_csv(os.path.join(OUTPUT_DIR, "cme_arrival_windows.csv"))
        cme_windows['cme_time'] = pd.to_datetime(cme_windows['cme_datetime'])
        cme_windows['arrival_start'] = pd.to_datetime(cme_windows['window_start'])
        cme_windows['arrival_end'] = pd.to_datetime(cme_windows['window_end'])
        
        print(f"✓ Main dataset: {len(df):,} data points")
        print(f"✓ CME events: {len(cme_events)} events")
        print(f"✓ CME windows: {len(cme_windows)} windows")
        
        return df, cme_events, cme_windows
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Please run step1_data_preparation.py first!")
        return None, None, None

def analyze_data_distribution(df):
    """Analyze the distribution of solar wind parameters"""
    print("\\nAnalyzing solar wind parameter distributions...")
    
    # Define parameters to analyze
    params = ['proton_density', 'proton_velocity', 'proton_temperature']
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Solar Wind Parameter Distributions', fontsize=16)
    
    # Distribution plots
    for i, param in enumerate(params):
        # Histogram
        axes[0, i].hist(df[param].dropna(), bins=100, alpha=0.7, density=True, color='steelblue')
        axes[0, i].set_title(f'{param.replace("_", " ").title()} Distribution')
        axes[0, i].set_xlabel(get_param_units(param))
        axes[0, i].set_ylabel('Density')
        axes[0, i].grid(True, alpha=0.3)
        
        # Box plot for CME vs non-CME
        cme_data = df[df['cme_event'] == 1][param].dropna()
        non_cme_data = df[df['cme_event'] == 0][param].dropna()
        
        box_data = [non_cme_data, cme_data]
        box_labels = ['Non-CME', 'CME']
        
        bp = axes[1, i].boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        axes[1, i].set_title(f'{param.replace("_", " ").title()}: CME vs Non-CME')
        axes[1, i].set_ylabel(get_param_units(param))
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step2_parameter_distributions.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    print("\\nStatistical Summary:")
    print("-" * 60)
    for param in params:
        print(f"\\n{param.replace('_', ' ').title()}:")
        print(f"  Overall: mean={df[param].mean():.2f}, std={df[param].std():.2f}")
        
        cme_stats = df[df['cme_event'] == 1][param]
        non_cme_stats = df[df['cme_event'] == 0][param]
        
        print(f"  CME: mean={cme_stats.mean():.2f}, std={cme_stats.std():.2f}")
        print(f"  Non-CME: mean={non_cme_stats.mean():.2f}, std={non_cme_stats.std():.2f}")
        
        # Simple t-test (qualitative)
        if cme_stats.mean() > non_cme_stats.mean():
            print(f"  → CME values are {cme_stats.mean()/non_cme_stats.mean():.2f}x higher on average")

def get_param_units(param):
    """Get units for parameters"""
    units = {
        'proton_density': 'Number Density (#/cm³)',
        'proton_velocity': 'Bulk Speed (km/s)',
        'proton_temperature': 'Thermal Velocity (km/s)'
    }
    return units.get(param, param)

def analyze_temporal_patterns(df, cme_events):
    """Analyze temporal patterns in the data"""
    print("\\nAnalyzing temporal patterns...")
    
    # Create time series plot
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle('Solar Wind Parameters Over Time', fontsize=16)
    
    params = ['proton_density', 'proton_velocity', 'proton_temperature']
    colors = ['blue', 'red', 'green']
    
    # Plot parameters
    for i, (param, color) in enumerate(zip(params, colors)):
        axes[i].plot(df['datetime'], df[param], color=color, alpha=0.6, linewidth=0.5)
        axes[i].set_ylabel(get_param_units(param))
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{param.replace("_", " ").title()} Time Series')
        
        # Mark CME events
        for _, cme in cme_events.iterrows():
            axes[i].axvline(x=cme['datetime'], color='red', linestyle='--', alpha=0.7)
    
    # Plot CME markers
    axes[3].scatter(df['datetime'], df['cme_event'], alpha=0.6, s=0.1, color='red')
    axes[3].set_ylabel('CME Marker')
    axes[3].set_xlabel('Time')
    axes[3].set_title('CME Event Markers')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step2_temporal_patterns.png"), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_correlations(df):
    """Analyze correlations between parameters"""
    print("\\nAnalyzing parameter correlations...")
    
    # Select numeric columns for correlation
    numeric_cols = ['proton_density', 'proton_velocity', 'proton_temperature', 
                   'proton_xvelocity', 'proton_yvelocity', 'proton_zvelocity']
    
    # Remove columns that don't exist in the dataframe
    existing_cols = [col for col in numeric_cols if col in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[existing_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Solar Wind Parameter Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step2_correlations.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\nStrong correlations (|r| > 0.5):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                param1 = corr_matrix.columns[i]
                param2 = corr_matrix.columns[j]
                print(f"  {param1} vs {param2}: r = {corr_val:.3f}")

def analyze_cme_signatures(df, cme_windows):
    """Analyze specific CME signatures"""
    print("\\nAnalyzing CME signatures...")
    
    # Create figure for CME signature analysis
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('CME Signature Analysis', fontsize=16)
    
    params = ['proton_density', 'proton_velocity', 'proton_temperature']
    
    # For each parameter, show before/during/after CME
    for idx, param in enumerate(params):
        # Time series around CME events
        ax1 = axes[idx, 0]
        ax2 = axes[idx, 1]
        
        # Collect data around CME windows
        before_cme = []
        during_cme = []
        after_cme = []
        
        for _, window in cme_windows.iterrows():
            # Define time windows
            cme_start = window['arrival_start']
            cme_end = window['arrival_end']
            before_start = cme_start - timedelta(hours=24)
            after_end = cme_end + timedelta(hours=24)
            
            # Get data for each period
            before_data = df[(df['datetime'] >= before_start) & 
                           (df['datetime'] < cme_start)][param].dropna()
            during_data = df[(df['datetime'] >= cme_start) & 
                           (df['datetime'] <= cme_end)][param].dropna()
            after_data = df[(df['datetime'] > cme_end) & 
                          (df['datetime'] <= after_end)][param].dropna()
            
            before_cme.extend(before_data.tolist())
            during_cme.extend(during_data.tolist())
            after_cme.extend(after_data.tolist())
        
        # Box plot comparison
        if before_cme and during_cme and after_cme:
            box_data = [before_cme, during_cme, after_cme]
            box_labels = ['Before CME\\n(24h)', 'During CME', 'After CME\\n(24h)']
            
            bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax1.set_title(f'{param.replace("_", " ").title()}: CME Periods')
            ax1.set_ylabel(get_param_units(param))
            ax1.grid(True, alpha=0.3)
            
            # Statistical summary
            stats_text = f"Before: μ={np.mean(before_cme):.2f}\\n"
            stats_text += f"During: μ={np.mean(during_cme):.2f}\\n"
            stats_text += f"After: μ={np.mean(after_cme):.2f}"
            ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, 
                    verticalalignment='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
            ax2.set_title(f'{param.replace("_", " ").title()}: Statistics')
            ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step2_cme_signatures.png"), dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_statistics(df, cme_events, cme_windows):
    """Generate comprehensive summary statistics"""
    print("\\nGenerating summary statistics...")
    
    # Data coverage summary
    print("\\nData Coverage Summary:")
    print("-" * 40)
    print(f"Time span: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
    print(f"Total data points: {len(df):,}")
    print(f"Data completeness: {len(df.dropna())/len(df)*100:.1f}%")
    
    # CME event summary
    print("\\nCME Event Summary:")
    print("-" * 40)
    print(f"Total CME events: {len(cme_events)}")
    print(f"CME periods (precise): {df['cme_event'].sum():,} points ({df['cme_event'].mean()*100:.2f}%)")
    print(f"CME periods (extended): {df['cme_event_extended'].sum():,} points ({df['cme_event_extended'].mean()*100:.2f}%)")
    
    # Parameter summary
    params = ['proton_density', 'proton_velocity', 'proton_temperature']
    print("\\nParameter Summary:")
    print("-" * 40)
    
    summary_stats = []
    for param in params:
        if param in df.columns:
            stats = {
                'Parameter': param.replace('_', ' ').title(),
                'Mean': f"{df[param].mean():.2f}",
                'Std': f"{df[param].std():.2f}",
                'Min': f"{df[param].min():.2f}",
                'Max': f"{df[param].max():.2f}",
                'CME_Mean': f"{df[df['cme_event']==1][param].mean():.2f}",
                'Enhancement': f"{df[df['cme_event']==1][param].mean()/df[df['cme_event']==0][param].mean():.2f}x"
            }
            summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False))
    
    # Save summary to file
    with open(os.path.join(OUTPUT_DIR, "step2_summary_statistics.txt"), 'w') as f:
        f.write("ISRO HACKATHON PS10 - STEP 2: EXPLORATORY DATA ANALYSIS SUMMARY\\n")
        f.write("="*80 + "\\n\\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("DATA COVERAGE SUMMARY:\\n")
        f.write("-" * 40 + "\\n")
        f.write(f"Time span: {df['datetime'].min()} to {df['datetime'].max()}\\n")
        f.write(f"Total duration: {(df['datetime'].max() - df['datetime'].min()).days} days\\n")
        f.write(f"Total data points: {len(df):,}\\n")
        f.write(f"Data completeness: {len(df.dropna())/len(df)*100:.1f}%\\n\\n")
        
        f.write("CME EVENT SUMMARY:\\n")
        f.write("-" * 40 + "\\n")
        f.write(f"Total CME events: {len(cme_events)}\\n")
        f.write(f"CME periods (precise): {df['cme_event'].sum():,} points ({df['cme_event'].mean()*100:.2f}%)\n")
        f.write(f"CME periods (extended): {df['cme_event_extended'].sum():,} points ({df['cme_event_extended'].mean()*100:.2f}%)\n\n")
        
        f.write("PARAMETER SUMMARY:\\n")
        f.write("-" * 40 + "\\n")
        f.write(summary_df.to_string(index=False))
        f.write("\\n\\n")
        
        f.write("KEY OBSERVATIONS:\\n")
        f.write("-" * 40 + "\\n")
        for param in params:
            if param in df.columns:
                enhancement = df[df['cme_event']==1][param].mean()/df[df['cme_event']==0][param].mean()
                if enhancement > 1.2:
                    f.write(f"• {param.replace('_', ' ').title()} shows {enhancement:.2f}x enhancement during CME events\\n")
                elif enhancement < 0.8:
                    f.write(f"• {param.replace('_', ' ').title()} shows {1/enhancement:.2f}x reduction during CME events\\n")
                else:
                    f.write(f"• {param.replace('_', ' ').title()} shows minimal change during CME events\\n")

def main():
    """Main execution function"""
    print("Starting exploratory data analysis...")
    
    # Load data
    df, cme_events, cme_windows = load_prepared_data()
    if df is None:
        return
    
    # Perform analyses
    print("\\n1. Analyzing data distributions...")
    analyze_data_distribution(df)
    
    print("\\n2. Analyzing temporal patterns...")
    analyze_temporal_patterns(df, cme_events)
    
    print("\\n3. Analyzing correlations...")
    analyze_correlations(df)
    
    print("\\n4. Analyzing CME signatures...")
    analyze_cme_signatures(df, cme_windows)
    
    print("\\n5. Generating summary statistics...")
    generate_summary_statistics(df, cme_events, cme_windows)
    
    print("\\n" + "="*80)
    print("STEP 2 COMPLETE: Exploratory Data Analysis finished successfully!")
    print("Key findings:")
    print("- Parameter distributions and CME vs non-CME comparisons")
    print("- Temporal patterns and correlations identified")
    print("- CME signature characteristics analyzed")
    print("- Summary statistics generated")
    print("\\nNext: Create step3_feature_engineering.py")
    print("="*80)

if __name__ == "__main__":
    main()
