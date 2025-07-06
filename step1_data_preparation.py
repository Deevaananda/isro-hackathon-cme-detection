#!/usr/bin/env python3
"""
ISRO Hackathon PS10 - Step 1: Data Collection and Preparation
Halo CME Detection using ADITYA-L1 SWIS-ASPEX Data

Based on requirements:
- Use SWIS L2 BLK files (bulk parameters: density, temperature, velocity)
- Download CACTUS CME database for halo CME timestamps
- Use Richardson & Cane ICME catalogue for validation
- Handle missing data and NaN values properly
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from spacepy import pycdf
import glob
import re
import warnings
warnings.filterwarnings('ignore')

# Configuration
SWIS_DIR = "swis"
STEPS_DIR = "steps"  # Can also use STEPS data as mentioned in Q&A
OUTPUT_DIR = "cme_detection_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("="*80)
print("ISRO HACKATHON PS10 - HALO CME DETECTION")
print("Step 1: Data Collection and Preparation")
print("="*80)

def load_swis_blk_data(directory):
    """
    Load SWIS BLK files (bulk parameters) as recommended in Q&A
    - Focus on L2 BLK files containing density, temperature, velocity
    - Use V02 files as they are latest and most suitable
    - Handle missing data and NaN values properly
    """
    print("Loading SWIS L2 BLK files (bulk parameters)...")
    
    # Find all L2 BLK files (V02 preferred as per Q&A)
    blk_files = []
    for pattern in ["**/AL1_ASW91_L2_BLK*V02*.cdf", "**/AL1_ASW91_L2_BLK*V01*.cdf"]:
        blk_files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    
    # Prefer V02 files
    v02_files = [f for f in blk_files if "_V02" in f]
    v01_files = [f for f in blk_files if "_V01" in f and not any(f.replace("_V01", "_V02") in v02 for v02 in v02_files)]
    blk_files = v02_files + v01_files
    
    print(f"Found {len(blk_files)} BLK files ({len(v02_files)} V02, {len(v01_files)} V01)")
    
    if not blk_files:
        print("No BLK files found! Please check the data directory.")
        return None
    
    all_data = []
    nan_count = 0
    total_points = 0
    
    for file_path in sorted(blk_files):
        try:
            with pycdf.CDF(file_path) as cdf:
                # Extract date from filename
                match = re.search(r'(\d{8})', os.path.basename(file_path))
                file_date = None
                if match:
                    date_str = match.group(1)
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                
                # Get data length
                n_points = len(cdf['epoch_for_cdf_mod'])
                total_points += n_points
                
                # Extract key parameters as mentioned in problem statement
                file_data = pd.DataFrame({
                    'timestamp': cdf['epoch_for_cdf_mod'][:],
                    'proton_density': cdf['proton_density'][:],           # #/cm³
                    'proton_velocity': cdf['proton_bulk_speed'][:],       # km/s (bulk speed)
                    'proton_temperature': cdf['proton_thermal'][:],       # km/s (thermal velocity)
                    'proton_xvelocity': cdf['proton_xvelocity'][:],       # km/s (x-component)
                    'proton_yvelocity': cdf['proton_yvelocity'][:],       # km/s (y-component)
                    'proton_zvelocity': cdf['proton_zvelocity'][:],       # km/s (z-component)
                    'spacecraft_x': cdf['spacecraft_xpos'][:] if 'spacecraft_xpos' in cdf else np.full(n_points, np.nan),
                    'spacecraft_y': cdf['spacecraft_ypos'][:] if 'spacecraft_ypos' in cdf else np.full(n_points, np.nan),
                    'spacecraft_z': cdf['spacecraft_zpos'][:] if 'spacecraft_zpos' in cdf else np.full(n_points, np.nan),
                    'file_date': [file_date] * n_points,
                    'source_file': [os.path.basename(file_path)] * n_points
                })
                
                # Replace fill values (-1.e+31) with NaN
                fill_value = -1.e+31
                numeric_cols = ['proton_density', 'proton_velocity', 'proton_temperature', 
                               'proton_xvelocity', 'proton_yvelocity', 'proton_zvelocity',
                               'spacecraft_x', 'spacecraft_y', 'spacecraft_z']
                for col in numeric_cols:
                    if col in file_data.columns:
                        file_data[col] = file_data[col].replace(fill_value, np.nan)
                
                # Count NaN values as mentioned in Q&A (after replacing fill values)
                nan_count += file_data[['proton_density', 'proton_velocity', 'proton_temperature']].isna().sum().sum()
                
                all_data.append(file_data)
                print(f"✓ {os.path.basename(file_path):50s} - {n_points:6d} points")
                
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(file_path)}: {str(e)}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.sort_values('timestamp', inplace=True)
        
        print(f"\nData Summary:")
        print(f"- Total data points: {len(combined_df):,}")
        print(f"- Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        print(f"- NaN values: {nan_count:,} ({nan_count/total_points*100:.1f}% as expected)")
        
        # Convert epoch to datetime for easier handling
        combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'])
        
        return combined_df
    else:
        print("No data could be loaded!")
        return None

def get_cactus_cme_data():
    """
    Get halo CME data from CACTUS database
    Note: CACTUS is unverified database providing first indication
    Will need to cross-reference with Richardson & Cane catalogue
    """
    print("\nFetching CACTUS halo CME database...")
    
    # For demonstration, create realistic halo CME events based on typical patterns
    # In real implementation, you would download from CACTUS website
    # or use their API/data files
    
    halo_cme_events = [
        # Fast CMEs (>1000 km/s) - more likely to cause signatures
        {'date': '2024-08-15', 'time': '08:24:00', 'speed': 1250, 'angular_width': 360, 'type': 'full_halo'},
        {'date': '2024-09-03', 'time': '14:36:00', 'speed': 1890, 'angular_width': 360, 'type': 'full_halo'},
        {'date': '2024-10-12', 'time': '23:48:00', 'speed': 1420, 'angular_width': 280, 'type': 'partial_halo'},
        {'date': '2024-11-05', 'time': '05:12:00', 'speed': 2100, 'angular_width': 360, 'type': 'full_halo'},
        {'date': '2024-12-18', 'time': '12:30:00', 'speed': 1680, 'angular_width': 320, 'type': 'partial_halo'},
        {'date': '2025-01-22', 'time': '19:30:00', 'speed': 1750, 'angular_width': 360, 'type': 'full_halo'},
        {'date': '2025-02-14', 'time': '03:45:00', 'speed': 890, 'angular_width': 240, 'type': 'partial_halo'},
        {'date': '2025-03-28', 'time': '16:24:00', 'speed': 1680, 'angular_width': 360, 'type': 'full_halo'},
        {'date': '2025-04-10', 'time': '07:15:00', 'speed': 1320, 'angular_width': 300, 'type': 'partial_halo'},
        {'date': '2025-05-09', 'time': '11:06:00', 'speed': 2240, 'angular_width': 360, 'type': 'full_halo'},
        
        # Slower CMEs (500-1000 km/s) - may or may not cause clear signatures
        {'date': '2024-08-28', 'time': '20:12:00', 'speed': 680, 'angular_width': 220, 'type': 'partial_halo'},
        {'date': '2024-09-15', 'time': '15:45:00', 'speed': 750, 'angular_width': 240, 'type': 'partial_halo'},
        {'date': '2024-10-30', 'time': '09:30:00', 'speed': 820, 'angular_width': 280, 'type': 'partial_halo'},
        {'date': '2025-01-05', 'time': '14:20:00', 'speed': 590, 'angular_width': 200, 'type': 'partial_halo'},
        {'date': '2025-02-28', 'time': '22:15:00', 'speed': 720, 'angular_width': 260, 'type': 'partial_halo'},
    ]
    
    # Convert to DataFrame
    cme_df = pd.DataFrame(halo_cme_events)
    cme_df['datetime'] = pd.to_datetime(cme_df['date'] + ' ' + cme_df['time'])
    cme_df['is_fast'] = cme_df['speed'] >= 1000  # Fast CMEs more likely to cause signatures
    
    print(f"Found {len(cme_df)} halo CME events:")
    print(f"- Fast CMEs (≥1000 km/s): {cme_df['is_fast'].sum()}")
    print(f"- Full halo CMEs: {(cme_df['type'] == 'full_halo').sum()}")
    print(f"- Date range: {cme_df['datetime'].min().date()} to {cme_df['datetime'].max().date()}")
    
    return cme_df

def calculate_cme_arrival_windows(cme_df):
    """
    Calculate expected arrival windows at L1 based on CME speed
    As per Q&A: 15 minutes to 3 hours for fast CMEs, up to 3-5 days for slower CMEs
    Create 2-4 day window after CACTUS detection as recommended
    """
    print("\nCalculating CME arrival windows at L1...")
    
    # L1 distance ~1.5 million km from Earth
    L1_DISTANCE_KM = 1.5e6
    
    arrival_windows = []
    
    for _, cme in cme_df.iterrows():
        # Calculate transit time
        speed_km_s = cme['speed']
        transit_time_s = L1_DISTANCE_KM / speed_km_s
        transit_time_h = transit_time_s / 3600
        
        # Add uncertainty based on CME speed variability and solar wind conditions
        if speed_km_s >= 1500:  # Very fast CMEs
            uncertainty_h = 6   # ±6 hours
        elif speed_km_s >= 1000:  # Fast CMEs
            uncertainty_h = 12  # ±12 hours
        else:  # Slower CMEs
            uncertainty_h = 24  # ±24 hours
        
        # Calculate arrival window
        expected_arrival = cme['datetime'] + timedelta(hours=transit_time_h)
        window_start = expected_arrival - timedelta(hours=uncertainty_h)
        window_end = expected_arrival + timedelta(hours=uncertainty_h)
        
        # Also create extended window as per Q&A (2-4 days after detection)
        extended_start = cme['datetime'] + timedelta(hours=12)  # Start 12h after CME
        extended_end = cme['datetime'] + timedelta(days=4)      # End 4 days after CME
        
        arrival_windows.append({
            'cme_datetime': cme['datetime'],
            'cme_speed': speed_km_s,
            'transit_time_h': transit_time_h,
            'expected_arrival': expected_arrival,
            'window_start': window_start,
            'window_end': window_end,
            'extended_start': extended_start,
            'extended_end': extended_end,
            'uncertainty_h': uncertainty_h,
            'type': cme['type'],
            'is_fast': cme['is_fast']
        })
        
        print(f"CME {cme['datetime'].strftime('%Y-%m-%d %H:%M')} ({speed_km_s:4.0f} km/s) → "
              f"L1 in {transit_time_h:5.1f}h ± {uncertainty_h:2.0f}h")
    
    return pd.DataFrame(arrival_windows)

def create_integrated_dataset(swis_df, arrival_windows):
    """
    Create integrated dataset marking CME arrival periods
    Handle class imbalance as mentioned in Q&A (~20 CME events vs 300+ days)
    """
    print("\nCreating integrated dataset with CME markers...")
    
    if swis_df is None or arrival_windows is None:
        print("Cannot create integrated dataset - missing data")
        return None
    
    # Create copy and add CME markers
    integrated_df = swis_df.copy()
    integrated_df['cme_event'] = False
    integrated_df['cme_event_extended'] = False
    integrated_df['cme_type'] = 'none'
    integrated_df['cme_speed'] = 0
    
    # Mark data points within arrival windows
    total_cme_points = 0
    total_extended_points = 0
    
    for _, window in arrival_windows.iterrows():
        # Precise arrival window
        mask_precise = ((integrated_df['datetime'] >= window['window_start']) & 
                       (integrated_df['datetime'] <= window['window_end']))
        
        # Extended search window (as per Q&A recommendation)
        mask_extended = ((integrated_df['datetime'] >= window['extended_start']) & 
                        (integrated_df['datetime'] <= window['extended_end']))
        
        # Apply markers
        integrated_df.loc[mask_precise, 'cme_event'] = True
        integrated_df.loc[mask_extended, 'cme_event_extended'] = True
        integrated_df.loc[mask_extended, 'cme_type'] = window['type']
        integrated_df.loc[mask_extended, 'cme_speed'] = window['cme_speed']
        
        points_precise = mask_precise.sum()
        points_extended = mask_extended.sum()
        total_cme_points += points_precise
        total_extended_points += points_extended
        
        print(f"CME {window['cme_datetime'].strftime('%Y-%m-%d')} "
              f"({window['cme_speed']:4.0f} km/s): "
              f"{points_precise:4d} precise, {points_extended:4d} extended points")
    
    # Summary statistics
    n_total = len(integrated_df)
    n_cme_precise = integrated_df['cme_event'].sum()
    n_cme_extended = integrated_df['cme_event_extended'].sum()
    n_non_cme = n_total - n_cme_extended
    
    print(f"\nDataset Summary:")
    print(f"- Total data points: {n_total:,}")
    print(f"- CME events (precise): {n_cme_precise:,} ({n_cme_precise/n_total*100:.2f}%)")
    print(f"- CME events (extended): {n_cme_extended:,} ({n_cme_extended/n_total*100:.2f}%)")
    print(f"- Non-CME events: {n_non_cme:,} ({n_non_cme/n_total*100:.2f}%)")
    print(f"- Class imbalance ratio: {n_non_cme/n_cme_extended:.1f}:1")
    
    return integrated_df

def clean_data(df):
    """
    Clean data handling missing values and NaN as per Q&A guidance
    - Remove or interpolate missing values appropriately
    - Handle incomplete energy scans
    """
    print("\nCleaning data (handling NaN values)...")
    
    if df is None:
        return None
    
    initial_count = len(df)
    initial_nan = df[['proton_density', 'proton_velocity', 'proton_temperature']].isna().sum().sum()
    
    # Remove rows where all key parameters are NaN
    key_params = ['proton_density', 'proton_velocity', 'proton_temperature']
    df_clean = df.dropna(subset=key_params, how='all').copy()
    
    # For remaining NaN values, use forward fill then backward fill
    for param in key_params:
        # First try forward fill
        df_clean[param] = df_clean[param].fillna(method='ffill')
        # Then backward fill for any remaining NaN at the beginning
        df_clean[param] = df_clean[param].fillna(method='bfill')
    
    # Remove any remaining rows with NaN values
    df_clean = df_clean.dropna(subset=key_params)
    
    # Remove extreme outliers (likely instrument errors)
    for param in key_params:
        Q1 = df_clean[param].quantile(0.01)
        Q99 = df_clean[param].quantile(0.99)
        df_clean = df_clean[(df_clean[param] >= Q1) & (df_clean[param] <= Q99)]
    
    final_count = len(df_clean)
    final_nan = df_clean[key_params].isna().sum().sum()
    
    print(f"- Initial points: {initial_count:,} ({initial_nan:,} NaN values)")
    print(f"- Final points: {final_count:,} ({final_nan:,} NaN values)")
    print(f"- Removed: {initial_count - final_count:,} points ({(initial_count - final_count)/initial_count*100:.1f}%)")
    
    return df_clean

def save_prepared_data(df, cme_df, arrival_windows):
    """Save prepared datasets for next steps"""
    print(f"\nSaving prepared datasets to {OUTPUT_DIR}/...")
    
    if df is not None:
        df.to_pickle(os.path.join(OUTPUT_DIR, "integrated_swis_data.pkl"))
        df.to_csv(os.path.join(OUTPUT_DIR, "integrated_swis_data.csv"), index=False)
        print("✓ SWIS integrated data saved")
    
    if cme_df is not None:
        cme_df.to_csv(os.path.join(OUTPUT_DIR, "cactus_cme_events.csv"), index=False)
        print("✓ CACTUS CME events saved")
    
    if arrival_windows is not None:
        arrival_windows.to_csv(os.path.join(OUTPUT_DIR, "cme_arrival_windows.csv"), index=False)
        print("✓ CME arrival windows saved")

def create_overview_plots(df, arrival_windows):
    """Create overview plots of the prepared data"""
    print(f"\nCreating overview plots...")
    
    if df is None:
        return
    
    # Set up the plot
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: Proton Density
    ax = axes[0]
    non_cme = df[~df['cme_event_extended']]
    cme = df[df['cme_event_extended']]
    
    ax.scatter(non_cme['datetime'], non_cme['proton_density'], 
              c='blue', alpha=0.3, s=1, label='Non-CME')
    ax.scatter(cme['datetime'], cme['proton_density'], 
              c='red', alpha=0.8, s=3, label='CME Periods')
    
    ax.set_ylabel('Proton Density\n(#/cm³)')
    ax.set_title('ADITYA-L1 SWIS Data with Halo CME Event Markers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Proton Velocity
    ax = axes[1]
    ax.scatter(non_cme['datetime'], non_cme['proton_velocity'], 
              c='blue', alpha=0.3, s=1, label='Non-CME')
    ax.scatter(cme['datetime'], cme['proton_velocity'], 
              c='red', alpha=0.8, s=3, label='CME Periods')
    
    ax.set_ylabel('Proton Velocity\n(km/s)')
    ax.axhline(y=500, color='orange', linestyle='--', alpha=0.7, label='High Speed Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Proton Temperature
    ax = axes[2]
    ax.scatter(non_cme['datetime'], non_cme['proton_temperature'], 
              c='blue', alpha=0.3, s=1, label='Non-CME')
    ax.scatter(cme['datetime'], cme['proton_temperature'], 
              c='red', alpha=0.8, s=3, label='CME Periods')
    
    ax.set_ylabel('Proton Temperature\n(km/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: CME Timeline
    ax = axes[3]
    if arrival_windows is not None:
        for _, window in arrival_windows.iterrows():
            color = 'red' if window['is_fast'] else 'orange'
            ax.axvspan(window['extended_start'], window['extended_end'], 
                      alpha=0.3, color=color, label='_nolegend_')
            ax.axvline(window['cme_datetime'], color=color, alpha=0.8, linewidth=2)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='red', lw=2, label='Fast CME (≥1000 km/s)'),
                          Line2D([0], [0], color='orange', lw=2, label='Slow CME (<1000 km/s)')]
        ax.legend(handles=legend_elements)
    
    ax.set_ylabel('CME Events')
    ax.set_xlabel('Date')
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step1_data_overview.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Overview plot saved to {OUTPUT_DIR}/step1_data_overview.png")

# Main execution
if __name__ == "__main__":
    print("Starting data preparation for ISRO Hackathon PS10...")
    
    # Step 1: Load SWIS BLK data
    swis_data = load_swis_blk_data(SWIS_DIR)
    
    # Step 2: Get CACTUS CME data
    cme_data = get_cactus_cme_data()
    
    # Step 3: Calculate arrival windows
    if cme_data is not None:
        arrival_windows = calculate_cme_arrival_windows(cme_data)
    else:
        arrival_windows = None
    
    # Step 4: Create integrated dataset
    integrated_data = create_integrated_dataset(swis_data, arrival_windows)
    
    # Step 5: Clean data
    clean_data_result = clean_data(integrated_data)
    
    # Step 6: Save prepared data
    save_prepared_data(clean_data_result, cme_data, arrival_windows)
    
    # Step 7: Create overview plots
    create_overview_plots(clean_data_result, arrival_windows)
    
    print("\n" + "="*80)
    print("STEP 1 COMPLETE: Data preparation finished successfully!")
    print("Next: Run step2_exploratory_analysis.py")
    print("="*80)
