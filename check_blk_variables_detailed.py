#!/usr/bin/env python3
"""
Check all variables in SWIS BLK files to understand the actual data structure
"""

import os
import glob
from spacepy import pycdf
import re

def check_blk_file_structure():
    """
    Check the structure of BLK files to understand available variables
    """
    swis_dir = "swis"
    
    # Find a few BLK files to check
    blk_files = glob.glob(os.path.join(swis_dir, "**/AL1_ASW91_L2_BLK*V02*.cdf"), recursive=True)
    
    if not blk_files:
        print("No BLK files found!")
        return
    
    # Check first 3 files
    for i, file_path in enumerate(blk_files[:3]):
        print(f"\n{'='*80}")
        print(f"FILE {i+1}: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        try:
            with pycdf.CDF(file_path) as cdf:
                print(f"CDF Info:")
                print(f"- File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
                
                # List all variables
                print(f"\nAll Variables ({len(cdf)} total):")
                for var_name in sorted(cdf.keys()):
                    var = cdf[var_name]
                    try:
                        shape = var.shape if hasattr(var, 'shape') else 'N/A'
                        dtype = var.dtype if hasattr(var, 'dtype') else 'N/A'
                        print(f"  {var_name:30s} - shape: {str(shape):15s} - dtype: {dtype}")
                    except Exception as e:
                        print(f"  {var_name:30s} - Error reading: {e}")
                
                # Check for epoch/time variables
                print(f"\nTime/Epoch Variables:")
                time_vars = [v for v in cdf.keys() if 'epoch' in v.lower() or 'time' in v.lower()]
                for var_name in time_vars:
                    try:
                        var = cdf[var_name]
                        print(f"  {var_name:30s} - length: {len(var):6d} - first few values:")
                        print(f"    {var[:3] if len(var) > 0 else 'No data'}")
                    except Exception as e:
                        print(f"  {var_name:30s} - Error: {e}")
                
                # Check for proton/particle variables
                print(f"\nProton/Particle Variables:")
                particle_vars = [v for v in cdf.keys() if any(word in v.lower() for word in ['proton', 'bulk', 'density', 'velocity', 'temperature', 'thermal', 'speed'])]
                for var_name in particle_vars:
                    try:
                        var = cdf[var_name]
                        shape = var.shape if hasattr(var, 'shape') else 'N/A'
                        dtype = var.dtype if hasattr(var, 'dtype') else 'N/A'
                        print(f"  {var_name:30s} - shape: {str(shape):15s} - dtype: {dtype}")
                        if len(var) > 0:
                            print(f"    Sample values: {var[:3]}")
                    except Exception as e:
                        print(f"  {var_name:30s} - Error: {e}")
                        
                # Check global attributes
                print(f"\nGlobal Attributes:")
                try:
                    attrs = cdf.attrs
                    for attr_name in sorted(attrs.keys())[:10]:  # Show first 10
                        print(f"  {attr_name:30s}: {attrs[attr_name]}")
                    if len(attrs) > 10:
                        print(f"  ... and {len(attrs) - 10} more attributes")
                except Exception as e:
                    print(f"  Error reading attributes: {e}")
                        
        except Exception as e:
            print(f"✗ Error opening {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    check_blk_file_structure()
