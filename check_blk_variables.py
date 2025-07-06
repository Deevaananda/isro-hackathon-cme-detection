#!/usr/bin/env python3
"""
Quick check of BLK file variables to fix the epoch issue
"""

import os
from spacepy import pycdf
import glob

def check_blk_file_variables():
    """Check what variables are actually available in BLK files"""
    
    # Find a V02 BLK file
    blk_files = glob.glob("swis/*L2_BLK*V02*.cdf")
    
    if not blk_files:
        print("No V02 BLK files found!")
        return
    
    # Check the first available file
    file_path = blk_files[0]
    print(f"Checking variables in: {os.path.basename(file_path)}")
    
    try:
        with pycdf.CDF(file_path) as cdf:
            print("\nAvailable variables:")
            for var_name in cdf.keys():
                var = cdf[var_name]
                print(f"  {var_name:30s} - Shape: {var.shape} - Type: {var.type()}")
                
            print("\nGlobal attributes:")
            for attr_name in cdf.attrs.keys():
                print(f"  {attr_name}: {cdf.attrs[attr_name]}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_blk_file_variables()
