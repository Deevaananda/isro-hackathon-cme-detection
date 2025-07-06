#!/usr/bin/env python3
"""
ISRO Hackathon PS10 - Step 3: Feature Engineering (Optimized Version)
Halo CME Detection using ADITYA-L1 SWIS-ASPEX Data

Optimized to prevent hanging during proton density processing
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = "cme_detection_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("="*80)
print("ISRO HACKATHON PS10 - HALO CME DETECTION")
print("Step 3: Feature Engineering (Optimized)")
print("="*80)

def load_prepared_data():
    """Load the prepared datasets from previous steps"""
    print("Loading prepared datasets...")
    
    try:
        # Load main dataset
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "integrated_swis_data.csv"))
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        print(f"✓ Dataset loaded: {len(df):,} data points")
        print(f"✓ Time range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        # Check for required columns
        required_cols = ['proton_density', 'proton_velocity', 'proton_temperature']
        available_cols = [col for col in required_cols if col in df.columns]
        print(f"✓ Available plasma parameters: {available_cols}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return None

def create_temporal_features(df, chunk_size=1000):
    """Create temporal features in chunks to prevent memory issues"""
    print("\n1. Creating temporal features...")
    
    # Define parameters to process
    params = ['proton_density', 'proton_velocity', 'proton_temperature']
    available_params = [p for p in params if p in df.columns]
    
    if not available_params:
        print("✗ No plasma parameters available for temporal features")
        return df
    
    print(f"   Processing parameters: {available_params}")
    
    # Create features in chunks
    total_rows = len(df)
    processed_rows = 0
    
    for param in available_params:
        print(f"   Processing {param}...")
        
        # Initialize arrays for new features
        ma_3h = np.full(total_rows, np.nan)
        ma_6h = np.full(total_rows, np.nan)
        gradient = np.full(total_rows, np.nan)
        
        # Process in chunks
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_data = df[param].iloc[start_idx:end_idx]
            
            # Moving averages (3-hour and 6-hour windows)
            window_3h = 36  # 3 hours at 5-min resolution
            window_6h = 72  # 6 hours at 5-min resolution
            
            if len(chunk_data) >= window_3h:
                ma_3h_chunk = chunk_data.rolling(window=window_3h, min_periods=1).mean()
                ma_3h[start_idx:end_idx] = ma_3h_chunk
            
            if len(chunk_data) >= window_6h:
                ma_6h_chunk = chunk_data.rolling(window=window_6h, min_periods=1).mean()
                ma_6h[start_idx:end_idx] = ma_6h_chunk
            
            # Gradient (rate of change)
            if len(chunk_data) > 1:
                gradient_chunk = np.gradient(chunk_data.fillna(0))
                gradient[start_idx:end_idx] = gradient_chunk
            
            processed_rows = end_idx
            if processed_rows % (chunk_size * 10) == 0:
                print(f"      Processed {processed_rows:,}/{total_rows:,} rows")
        
        # Add features to dataframe
        df[f'{param}_ma_3h'] = ma_3h
        df[f'{param}_ma_6h'] = ma_6h
        df[f'{param}_gradient'] = gradient
        
        print(f"   ✓ {param} temporal features created")
    
    print("✓ Temporal features completed")
    return df

def create_physics_features(df):
    """Create physics-based derived parameters"""
    print("\n2. Creating physics-based features...")
    
    feature_count = 0
    
    # Kinetic energy proxy
    if 'proton_density' in df.columns and 'proton_velocity' in df.columns:
        print("   Creating kinetic energy proxy...")
        df['kinetic_energy_proxy'] = df['proton_density'] * df['proton_velocity']**2
        feature_count += 1
    
    # Dynamic pressure
    if 'proton_density' in df.columns and 'proton_velocity' in df.columns:
        print("   Creating dynamic pressure...")
        df['dynamic_pressure'] = df['proton_density'] * df['proton_velocity']**2 * 1.67e-27 * 1e6
        feature_count += 1
    
    # Thermal speed
    if 'proton_temperature' in df.columns:
        print("   Creating thermal speed...")
        df['thermal_speed'] = np.sqrt(2 * 1.38e-23 * df['proton_temperature'] / 1.67e-27)
        feature_count += 1
    
    # Density enhancement
    if 'proton_density' in df.columns:
        print("   Creating density enhancement...")
        df['density_enhancement'] = df['proton_density'] / 5.0  # Normalized to typical solar wind
        feature_count += 1
    
    # Temperature ratio
    if 'proton_temperature' in df.columns:
        print("   Creating temperature ratio...")
        df['temperature_ratio'] = df['proton_temperature'] / 100000  # Normalized to 100,000 K
        feature_count += 1
    
    print(f"✓ Created {feature_count} physics-based features")
    return df

def create_anomaly_features(df, chunk_size=1000):
    """Create anomaly detection features"""
    print("\n3. Creating anomaly detection features...")
    
    params = ['proton_density', 'proton_velocity', 'proton_temperature']
    available_params = [p for p in params if p in df.columns]
    
    if not available_params:
        print("✗ No parameters available for anomaly detection")
        return df
    
    total_rows = len(df)
    
    for param in available_params:
        print(f"   Processing anomalies for {param}...")
        
        # Z-score based anomaly detection
        z_scores = np.full(total_rows, np.nan)
        
        # Process in chunks
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_data = df[param].iloc[start_idx:end_idx]
            
            # Calculate z-scores for chunk
            if len(chunk_data.dropna()) > 1:
                chunk_z = zscore(chunk_data.fillna(chunk_data.mean()))
                z_scores[start_idx:end_idx] = chunk_z
        
        df[f'{param}_zscore'] = z_scores
        df[f'{param}_anomaly'] = (np.abs(z_scores) > 2.5).astype(int)
    
    print("✓ Anomaly detection features completed")
    return df

def create_composite_features(df):
    """Create composite CME detection features"""
    print("\n4. Creating composite CME detection features...")
    
    feature_count = 0
    
    # Combined CME score
    score_components = []
    
    if 'proton_velocity' in df.columns:
        velocity_norm = (df['proton_velocity'] - 400) / 100  # Normalized velocity enhancement
        score_components.append(velocity_norm.clip(0, 5))
        feature_count += 1
    
    if 'proton_density' in df.columns:
        density_norm = (df['proton_density'] - 5) / 5  # Normalized density enhancement
        score_components.append(density_norm.clip(0, 3))
        feature_count += 1
    
    if 'proton_temperature' in df.columns:
        temp_norm = (df['proton_temperature'] - 100000) / 50000  # Normalized temperature
        score_components.append(temp_norm.clip(0, 2))
        feature_count += 1
    
    if score_components:
        df['cme_composite_score'] = np.sum(score_components, axis=0)
        feature_count += 1
    
    print(f"✓ Created {feature_count} composite features")
    return df

def save_engineered_features(df):
    """Save the enhanced dataset with engineered features"""
    print("\n5. Saving engineered features...")
    
    # Save enhanced dataset
    output_file = os.path.join(OUTPUT_DIR, "enhanced_swis_data.csv")
    df.to_csv(output_file, index=False)
    print(f"✓ Enhanced dataset saved: {output_file}")
    
    # Save feature summary
    feature_cols = [col for col in df.columns if any(suffix in col for suffix in 
                   ['_ma_3h', '_ma_6h', '_gradient', '_zscore', '_anomaly', 'kinetic_energy', 
                    'dynamic_pressure', 'thermal_speed', 'density_enhancement', 'temperature_ratio', 
                    'cme_composite_score'])]
    
    summary_file = os.path.join(OUTPUT_DIR, "engineered_features_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("ENGINEERED FEATURES SUMMARY\\n")
        f.write("="*50 + "\\n\\n")
        f.write(f"Total data points: {len(df):,}\\n")
        f.write(f"Original features: {len(df.columns) - len(feature_cols)}\\n")
        f.write(f"Engineered features: {len(feature_cols)}\\n\\n")
        
        f.write("Engineered Features:\\n")
        f.write("-" * 30 + "\\n")
        for i, col in enumerate(feature_cols, 1):
            f.write(f"{i:2d}. {col}\\n")
    
    print(f"✓ Feature summary saved: {summary_file}")
    
    return len(feature_cols)

def create_feature_visualization(df):
    """Create visualization of engineered features"""
    print("\n6. Creating feature visualization...")
    
    try:
        # Get feature columns
        feature_cols = [col for col in df.columns if any(suffix in col for suffix in 
                       ['_ma_3h', '_ma_6h', '_gradient', '_zscore', 'cme_composite_score'])]
        
        if len(feature_cols) == 0:
            print("✗ No features available for visualization")
            return
        
        # Create subplot grid
        n_features = min(9, len(feature_cols))  # Show max 9 features
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # Sample data for faster plotting
        sample_size = min(5000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        for i, col in enumerate(feature_cols[:n_features]):
            ax = axes[i]
            
            # Plot feature distribution
            data = sample_df[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{col}', fontsize=10)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        
        # Hide unused subplots
        for i in range(n_features, 9):
            axes[i].set_visible(False)
        
        plt.suptitle('Engineered Features Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "step3_engineered_features.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Feature visualization saved")
        
    except Exception as e:
        print(f"✗ Error creating visualization: {str(e)}")

def main():
    """Main feature engineering pipeline"""
    print("Starting optimized feature engineering pipeline...")
    
    # Load data
    df = load_prepared_data()
    if df is None:
        print("✗ Failed to load data. Exiting.")
        return
    
    initial_features = len(df.columns)
    print(f"Initial features: {initial_features}")
    
    # Feature engineering steps
    df = create_temporal_features(df, chunk_size=500)  # Smaller chunks
    df = create_physics_features(df)
    df = create_anomaly_features(df, chunk_size=500)  # Smaller chunks
    df = create_composite_features(df)
    
    # Save results
    engineered_features = save_engineered_features(df)
    create_feature_visualization(df)
    
    print(f"\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print(f"• Initial features: {initial_features}")
    print(f"• Engineered features: {engineered_features}")
    print(f"• Total features: {len(df.columns)}")
    print(f"• Output directory: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
