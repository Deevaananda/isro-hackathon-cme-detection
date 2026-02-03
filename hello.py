#!/usr/bin/env python3
"""
ISRO Hackathon PS10 - Hello Functionality
Simple greeting system for the CME Detection Project
"""

import sys
from datetime import datetime

def say_hello(name=None):
    """
    Simple greeting function for the ISRO CME Detection System
    
    Args:
        name (str, optional): Name to greet. Defaults to None.
    
    Returns:
        str: Greeting message
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if name:
        message = f"Hello {name}! Welcome to the ISRO Hackathon PS10 CME Detection System."
    else:
        message = "Hello! Welcome to the ISRO Hackathon PS10 CME Detection System."
    
    full_message = f"""
{message}
Time: {current_time}
Mission: Halo CME Detection using ADITYA-L1 SWIS-ASPEX Data
Status: System Ready ✓
"""
    return full_message

def main():
    """Main function for hello script"""
    print("=" * 70)
    print("ISRO HACKATHON PS10 - CME DETECTION SYSTEM")
    print("=" * 70)
    
    # Check if a name was provided as command line argument
    name = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Print the greeting
    print(say_hello(name))
    
    print("🚀 Ready for space weather analysis!")
    print("=" * 70)

if __name__ == "__main__":
    main()