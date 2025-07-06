#!/usr/bin/env python3
"""
ISRO Hackathon PS10 - Step 3: Feature Engineering
Halo CME Detection using ADITYA-L1 SWIS-ASPEX Data

Based on requirements:
- Derive new time-series features (moving averages, gradients, combined metrics)
- Create physics-based derived parameters
- En        plt.tight_layout()
    plt.        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "step3_feature_overview.png"), dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show to prevent hangingefig(os.path.join(OUTPUT_DIR, "step3_feature_correlations.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to prevent hanging.tight_layout()
    plt.        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "step3_engineered_features.png"), dpi=300, bbox_inches='tight')
        # plt.show()  # Disabled to prevent hangingefig(os.path.join(OUTPUT_DIR, "step3_feature_importance.png"), dpi=300, bbox_inches='tight')
    # plt.show()  # Disabled to prevent hangingeer features for CME detection
- Handle temporal dependencies and anomaly detection
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
print("Step 3: Feature Engineering")
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
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Please run step1_data_preparation.py first!")
        return None

def create_temporal_features(df, window_sizes=[12, 60, 288, 1440]):
    """
    Create temporal features using moving windows
    
    Args:
        window_sizes: List of window sizes in data points
                     [12=1h, 60=5h, 288=24h, 1440=5days] assuming 5min cadence
    """
    print("\\nCreating temporal features...")
    
    params = ['proton_density', 'proton_velocity', 'proton_temperature']
    
    # Sort by datetime to ensure proper time series operations
    df = df.sort_values('datetime').reset_index(drop=True)
    
    for param in params:
        if param not in df.columns:
            continue
            
        print(f"  Processing {param}...")
        
        # Moving averages
        for window in window_sizes:
            window_name = get_window_name(window)
            df[f'{param}_ma_{window_name}'] = df[param].rolling(window=window, min_periods=1).mean()
            df[f'{param}_std_{window_name}'] = df[param].rolling(window=window, min_periods=1).std()
        
        # Gradients (rate of change)
        df[f'{param}_gradient_1h'] = df[param].diff(periods=12)  # 1-hour gradient
        df[f'{param}_gradient_6h'] = df[param].diff(periods=72)  # 6-hour gradient
        
        # Relative changes
        df[f'{param}_pct_change_1h'] = df[param].pct_change(periods=12)
        df[f'{param}_pct_change_6h'] = df[param].pct_change(periods=72)
        
        # Z-scores (anomaly detection)
        df[f'{param}_zscore_1h'] = df[param].rolling(window=12).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
        df[f'{param}_zscore_24h'] = df[param].rolling(window=288).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
    
    print(f"  ✓ Created temporal features for {len(params)} parameters")
    return df

def get_window_name(window_size):
    """Convert window size to readable name"""
    if window_size <= 12:
        return f"{window_size}p"  # points
    elif window_size <= 60:
        return f"{window_size//12}h"  # hours
    elif window_size <= 288:
        return f"{window_size//12}h"  # hours
    else:
        return f"{window_size//288}d"  # days

def create_physics_features(df):
    """Create physics-based derived parameters"""
    print("\\nCreating physics-based features...")
    
    # Kinetic energy proxy (mass * velocity^2)
    if 'proton_density' in df.columns and 'proton_velocity' in df.columns:
        df['kinetic_energy_proxy'] = df['proton_density'] * df['proton_velocity']**2
        print("  ✓ Kinetic energy proxy")
    
    # Dynamic pressure (density * velocity^2)
    if 'proton_density' in df.columns and 'proton_velocity' in df.columns:
        df['dynamic_pressure'] = df['proton_density'] * df['proton_velocity']**2 * 1.67e-27 * 1e6  # Convert to SI-like units
        print("  ✓ Dynamic pressure")
    
    # Thermal speed ratio
    if 'proton_temperature' in df.columns and 'proton_velocity' in df.columns:
        df['thermal_velocity_ratio'] = df['proton_temperature'] / df['proton_velocity']
        print("  ✓ Thermal velocity ratio")
    
    # Speed ratio to quiet solar wind (400 km/s baseline)
    if 'proton_velocity' in df.columns:
        df['velocity_enhancement'] = df['proton_velocity'] / 400.0
        print("  ✓ Velocity enhancement factor")
    
    # Density ratio to quiet solar wind (5 /cm³ baseline)
    if 'proton_density' in df.columns:
        df['density_enhancement'] = df['proton_density'] / 5.0
        print("  ✓ Density enhancement factor")
    
    # Combined enhancement metric
    if 'velocity_enhancement' in df.columns and 'density_enhancement' in df.columns:
        df['combined_enhancement'] = np.sqrt(df['velocity_enhancement']**2 + df['density_enhancement']**2)
        print("  ✓ Combined enhancement metric")
    
    # Total velocity magnitude
    if all(col in df.columns for col in ['proton_xvelocity', 'proton_yvelocity', 'proton_zvelocity']):
        df['total_velocity_magnitude'] = np.sqrt(
            df['proton_xvelocity']**2 + 
            df['proton_yvelocity']**2 + 
            df['proton_zvelocity']**2
        )
        print("  ✓ Total velocity magnitude")
    
    return df

def create_anomaly_features(df):
    """Create anomaly detection features"""
    print("\\nCreating anomaly detection features...")
    
    params = ['proton_density', 'proton_velocity', 'proton_temperature']
    
    for param in params:
        if param not in df.columns:
            continue
            
        # Local outlier detection using IQR
        q75 = df[param].quantile(0.75)
        q25 = df[param].quantile(0.25)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        df[f'{param}_is_outlier'] = ((df[param] < lower_bound) | (df[param] > upper_bound)).astype(int)
        df[f'{param}_outlier_score'] = np.maximum(
            (lower_bound - df[param]) / iqr,
            (df[param] - upper_bound) / iqr
        ).fillna(0).clip(lower=0)
        
        # Rolling percentile features
        df[f'{param}_percentile_1h'] = df[param].rolling(window=12).rank(pct=True)
        df[f'{param}_percentile_24h'] = df[param].rolling(window=288).rank(pct=True)
    
    print(f"  ✓ Created anomaly features for {len(params)} parameters")
    return df

def create_composite_features(df):
    """Create composite features combining multiple parameters"""
    print("\\nCreating composite features...")
    
    # Multi-parameter enhancement score
    enhancement_cols = [col for col in df.columns if 'enhancement' in col and col != 'combined_enhancement']
    if len(enhancement_cols) >= 2:
        df['multi_enhancement_score'] = df[enhancement_cols].mean(axis=1)
        print("  ✓ Multi-parameter enhancement score")
    
    # Anomaly count (how many parameters are anomalous)
    outlier_cols = [col for col in df.columns if '_is_outlier' in col]
    if outlier_cols:
        df['total_anomaly_count'] = df[outlier_cols].sum(axis=1)
        print("  ✓ Total anomaly count")
    
    # Combined Z-score magnitude
    zscore_1h_cols = [col for col in df.columns if '_zscore_1h' in col]
    if zscore_1h_cols:
        df['combined_zscore_1h'] = np.sqrt((df[zscore_1h_cols]**2).sum(axis=1))
        print("  ✓ Combined Z-score (1h)")
    
    zscore_24h_cols = [col for col in df.columns if '_zscore_24h' in col]
    if zscore_24h_cols:
        df['combined_zscore_24h'] = np.sqrt((df[zscore_24h_cols]**2).sum(axis=1))
        print("  ✓ Combined Z-score (24h)")
    
    # Event probability score (combination of multiple indicators)
    if all(col in df.columns for col in ['combined_enhancement', 'total_anomaly_count', 'combined_zscore_1h']):
        # Normalize and combine different metrics
        df['event_probability_score'] = (
            0.4 * (df['combined_enhancement'] - 1).clip(lower=0) +
            0.3 * df['total_anomaly_count'] / len(outlier_cols) +
            0.3 * df['combined_zscore_1h'].clip(upper=5) / 5
        )
        print("  ✓ Event probability score")
    
    return df

def analyze_feature_importance(df):
    """Analyze which features are most correlated with CME events"""
    print("\\nAnalyzing feature importance...")
    
    # Get all engineered features
    original_cols = ['proton_density', 'proton_velocity', 'proton_temperature', 
                    'proton_xvelocity', 'proton_yvelocity', 'proton_zvelocity']
    engineered_cols = [col for col in df.columns if col not in original_cols and 
                      col not in ['timestamp', 'datetime', 'cme_event', 'cme_event_extended', 
                                 'cme_type', 'cme_speed', 'file_date', 'source_file', 
                                 'spacecraft_x', 'spacecraft_y', 'spacecraft_z']]
    
    # Calculate correlations with CME events
    correlations = []
    for col in engineered_cols:
        if df[col].dtype in ['int64', 'float64']:
            try:
                corr = df[col].corr(df['cme_event'])
                if not np.isnan(corr):
                    correlations.append({
                        'feature': col,
                        'correlation': abs(corr),
                        'correlation_signed': corr
                    })
            except:
                pass
    
    # Sort by absolute correlation
    correlations_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    
    print(f"\\nTop 20 features by correlation with CME events:")
    print("-" * 70)
    top_features = correlations_df.head(20)
    for _, row in top_features.iterrows():
        print(f"{row['feature']:50s} {row['correlation_signed']:8.4f}")
    
    # Save to file
    correlations_df.to_csv(os.path.join(OUTPUT_DIR, "step3_feature_importance.csv"), index=False)
    
    return correlations_df

def create_feature_visualizations(df, top_features):
    """Create visualizations for top features"""
    print("\\nCreating feature visualizations...")
    
    # Select top 8 features for visualization
    top_8_features = top_features.head(8)['feature'].tolist()
    
    # Create feature distribution plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_8_features):
        if feature in df.columns:
            # Plot distributions for CME vs non-CME
            cme_data = df[df['cme_event'] == 1][feature].dropna()
            non_cme_data = df[df['cme_event'] == 0][feature].dropna()
            
            if len(cme_data) > 0 and len(non_cme_data) > 0:
                axes[i].hist(non_cme_data, bins=50, alpha=0.6, label='Non-CME', density=True, color='blue')
                axes[i].hist(cme_data, bins=50, alpha=0.6, label='CME', density=True, color='red')
                axes[i].set_title(f'{feature}', fontsize=10)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Top Engineered Features: CME vs Non-CME Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step3_top_features.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create time series plot of top feature
    if len(top_8_features) > 0:
        top_feature = top_8_features[0]
        
        plt.figure(figsize=(16, 8))
        
        # Plot the top feature over time
        plt.subplot(2, 1, 1)
        plt.plot(df['datetime'], df[top_feature], color='blue', alpha=0.7, linewidth=0.5)
        plt.title(f'Time Series: {top_feature}')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        # Mark CME events
        cme_times = df[df['cme_event'] == 1]['datetime']
        for cme_time in cme_times:
            plt.axvline(x=cme_time, color='red', alpha=0.5, linewidth=0.5)
        
        # Plot CME markers
        plt.subplot(2, 1, 2)
        plt.scatter(df['datetime'], df['cme_event'], alpha=0.6, s=0.1, color='red')
        plt.title('CME Event Markers')
        plt.ylabel('CME Event')
        plt.xlabel('Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "step3_top_feature_timeseries.png"), dpi=300, bbox_inches='tight')
        plt.show()

def save_engineered_dataset(df):
    """Save the dataset with engineered features"""
    print("\\nSaving engineered dataset...")
    
    # Get feature counts
    original_cols = ['proton_density', 'proton_velocity', 'proton_temperature', 
                    'proton_xvelocity', 'proton_yvelocity', 'proton_zvelocity']
    engineered_cols = [col for col in df.columns if col not in original_cols and 
                      col not in ['timestamp', 'datetime', 'cme_event', 'cme_event_extended', 
                                 'cme_type', 'cme_speed', 'file_date', 'source_file', 
                                 'spacecraft_x', 'spacecraft_y', 'spacecraft_z']]
    
    print(f"  Original features: {len(original_cols)}")
    print(f"  Engineered features: {len(engineered_cols)}")
    print(f"  Total features: {len(df.columns)}")
    
    # Save as CSV and pickle
    output_path_csv = os.path.join(OUTPUT_DIR, "engineered_swis_data.csv")
    output_path_pkl = os.path.join(OUTPUT_DIR, "engineered_swis_data.pkl")
    
    df.to_csv(output_path_csv, index=False)
    df.to_pickle(output_path_pkl)
    
    print(f"  ✓ Saved to {output_path_csv}")
    print(f"  ✓ Saved to {output_path_pkl}")
    
    # Create feature summary
    feature_summary = {
        'total_features': len(df.columns),
        'original_features': len(original_cols),
        'engineered_features': len(engineered_cols),
        'data_points': len(df),
        'cme_points': df['cme_event'].sum(),
        'cme_percentage': df['cme_event'].mean() * 100
    }
    
    # Save feature list
    feature_info = pd.DataFrame({
        'feature_name': df.columns,
        'feature_type': ['original' if col in original_cols else 'engineered' if col in engineered_cols else 'metadata' for col in df.columns],
        'dtype': [str(df[col].dtype) for col in df.columns],
        'null_count': [df[col].isnull().sum() for col in df.columns],
        'null_percentage': [df[col].isnull().sum() / len(df) * 100 for col in df.columns]
    })
    
    feature_info.to_csv(os.path.join(OUTPUT_DIR, "step3_feature_summary.csv"), index=False)
    
    return feature_summary

def main():
    """Main execution function"""
    print("Starting feature engineering...")
    
    # Load data
    df = load_prepared_data()
    if df is None:
        return
    
    print(f"\\nInitial dataset shape: {df.shape}")
    
    # Create different types of features
    print("\\n1. Creating temporal features...")
    df = create_temporal_features(df)
    
    print("\\n2. Creating physics-based features...")
    df = create_physics_features(df)
    
    print("\\n3. Creating anomaly detection features...")
    df = create_anomaly_features(df)
    
    print("\\n4. Creating composite features...")
    df = create_composite_features(df)
    
    print(f"\\nFinal dataset shape: {df.shape}")
    print(f"Added {df.shape[1] - len(pd.read_csv(os.path.join(OUTPUT_DIR, 'integrated_swis_data.csv')).columns)} new features")
    
    # Analyze feature importance
    print("\\n5. Analyzing feature importance...")
    top_features = analyze_feature_importance(df)
    
    # Create visualizations
    print("\\n6. Creating visualizations...")
    create_feature_visualizations(df, top_features)
    
    # Save engineered dataset
    print("\\n7. Saving engineered dataset...")
    summary = save_engineered_dataset(df)
    
    print("\\n" + "="*80)
    print("STEP 3 COMPLETE: Feature Engineering finished successfully!")
    print("\\nSummary:")
    print(f"- Total features: {summary['total_features']}")
    print(f"- Original features: {summary['original_features']}")
    print(f"- Engineered features: {summary['engineered_features']}")
    print(f"- Data points: {summary['data_points']:,}")
    print(f"- CME events: {summary['cme_points']:,} ({summary['cme_percentage']:.2f}%)")
    print("\\nKey feature types created:")
    print("- Temporal: Moving averages, gradients, Z-scores")
    print("- Physics: Dynamic pressure, enhancement factors, composite metrics")
    print("- Anomaly: Outlier detection, percentiles, anomaly counts")
    print("- Composite: Multi-parameter scores, probability estimates")
    print("\\nNext: Create step4_threshold_determination.py")
    print("="*80)

if __name__ == "__main__":
    main()
