#!/usr/bin/env python3
"""
Time-Difference Propagation Visualization
Shows the critical time gap between CME detection and arrival
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CME TIME-DIFFERENCE PROPAGATION MODELING\nSun Detection → L1 Arrival (1-5 Day Gap)', 
             fontsize=16, fontweight='bold')

# 1. Propagation Time vs Velocity
velocities = np.linspace(200, 2000, 100)
basic_times = 1.5e6 / velocities / 3600  # Basic transit time in hours

# Apply corrections
corrected_times = []
for v in velocities:
    if v > 1000:
        correction = 1.2 + 0.1 * (v - 1000) / 500
    elif v < 400:
        correction = 0.85 - 0.05 * (400 - v) / 100
    else:
        correction = 1.0
    corrected_times.append(basic_times[list(velocities).index(v)] * correction)

ax1.plot(velocities, basic_times, 'b--', label='Basic Transit Time', linewidth=2)
ax1.plot(velocities, corrected_times, 'r-', label='Physics-Corrected Time', linewidth=3)
ax1.axhline(y=24, color='gray', linestyle=':', alpha=0.7, label='1 Day')
ax1.axhline(y=72, color='gray', linestyle=':', alpha=0.7, label='3 Days')
ax1.axhline(y=120, color='gray', linestyle=':', alpha=0.7, label='5 Days')
ax1.set_xlabel('CME Velocity (km/s)')
ax1.set_ylabel('Propagation Time (hours)')
ax1.set_title('A) Velocity-Dependent Propagation Times')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(200, 2000)
ax1.set_ylim(0, 150)

# 2. Timeline showing detection to arrival
timeline_data = [
    {'name': 'SOHO/LASCO\nDetection', 'start': 0, 'duration': 1, 'color': 'orange'},
    {'name': 'Propagation\n(1-5 days)', 'start': 1, 'duration': 72, 'color': 'lightblue'},
    {'name': 'L1 Arrival\nWindow', 'start': 73, 'duration': 24, 'color': 'red'},
    {'name': 'SWIS-ASPEX\nMeasurement', 'start': 85, 'duration': 1, 'color': 'green'}
]

y_pos = [3, 2, 1, 0]
for i, item in enumerate(timeline_data):
    ax2.barh(y_pos[i], item['duration'], left=item['start'], 
             color=item['color'], alpha=0.7, height=0.5)
    ax2.text(item['start'] + item['duration']/2, y_pos[i], item['name'], 
             ha='center', va='center', fontweight='bold', fontsize=9)

ax2.set_xlim(0, 100)
ax2.set_ylim(-0.5, 3.5)
ax2.set_xlabel('Time (hours from CME detection)')
ax2.set_title('B) CME Detection to L1 Arrival Timeline')
ax2.set_yticks([])
ax2.grid(True, alpha=0.3, axis='x')

# 3. Example velocity corrections
cme_types = ['Very Fast\n(>1500 km/s)', 'Fast\n(1000-1500)', 'Medium\n(500-1000)', 
             'Slow\n(300-500)', 'Very Slow\n(<300)']
basic_times_examples = [17, 25, 50, 100, 150]
corrected_times_examples = [22, 30, 50, 85, 130]

x_pos = np.arange(len(cme_types))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, basic_times_examples, width, 
                label='Basic Transit', color='lightblue', alpha=0.8)
bars2 = ax3.bar(x_pos + width/2, corrected_times_examples, width,
                label='Physics-Corrected', color='orange', alpha=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}h', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}h', ha='center', va='bottom', fontsize=9)

ax3.set_xlabel('CME Type')
ax3.set_ylabel('Propagation Time (hours)')
ax3.set_title('C) Physics Corrections by CME Type')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(cme_types, fontsize=9)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Uncertainty and arrival windows
days = np.arange(1, 8)
central_prediction = [24, 30, 40, 60, 85, 110, 140]
upper_bound = [36, 42, 52, 72, 97, 122, 152]
lower_bound = [12, 18, 28, 48, 73, 98, 128]

ax4.fill_between(days, lower_bound, upper_bound, alpha=0.3, color='red', 
                 label='Uncertainty Window (±12h)')
ax4.plot(days, central_prediction, 'ro-', linewidth=2, markersize=6,
         label='Central Prediction')

ax4.set_xlabel('Days from Sun Detection')
ax4.set_ylabel('Expected Arrival Time (hours)')
ax4.set_title('D) Arrival Time Predictions with Uncertainty')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0.5, 7.5)

plt.tight_layout()
plt.savefig('cme_detection_output/time_difference_propagation_visualization.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("✓ Time-difference propagation visualization saved!")
print("  File: cme_detection_output/time_difference_propagation_visualization.png")
