#!/usr/bin/env python3
"""
Create an installer background image for the WeightTracker DMG
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_installer_background():
    """Create a professional installer background image"""
    
    # Create figure with proper dimensions (800x500 for installer)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    
    # Set background color (light blue-gray)
    ax.set_facecolor('#f0f4f8')
    fig.patch.set_facecolor('#f0f4f8')
    
    # Add gradient effect
    gradient = np.linspace(0, 1, 100)
    for i, alpha in enumerate(gradient):
        y_pos = 5 - (i * 0.05)
        if y_pos >= 0:
            ax.axhline(y=y_pos, color='white', alpha=alpha * 0.3, linewidth=0.5)
    
    # Add title
    ax.text(4, 4.5, 'BodyMetrics', fontsize=24, fontweight='bold', 
            ha='center', va='center', color='#2c3e50')
    
    # Add subtitle
    ax.text(4, 4.1, 'Professional Fitness Tracking & Analytics', fontsize=14, 
            ha='center', va='center', color='#34495e', style='italic')
    
    # Add installation instructions
    instructions = [
        "1. Drag WeightTracker.app to Applications",
        "2. Launch from Applications folder",
        "3. Start tracking your fitness journey!"
    ]
    
    for i, instruction in enumerate(instructions):
        y_pos = 3.2 - (i * 0.4)
        ax.text(4, y_pos, instruction, fontsize=12, ha='center', va='center', 
                color='#2c3e50', weight='medium')
    
    # Add feature highlights
    features = [
        "📊 Advanced Analytics",
        "📈 Trend Analysis", 
        "🎯 Goal Tracking",
        "💾 Data Export"
    ]
    
    for i, feature in enumerate(features):
        x_pos = 1.5 + (i * 1.5)
        y_pos = 1.8
        ax.text(x_pos, y_pos, feature, fontsize=10, ha='center', va='center', 
                color='#34495e')
    
    # Add decorative elements
    # Left side - app icon placeholder
    left_icon = FancyBboxPatch((0.8, 0.8), 1.2, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#3498db', edgecolor='#2980b9', linewidth=2)
    ax.add_patch(left_icon)
    ax.text(1.4, 1.4, 'WT', fontsize=16, ha='center', va='center', 
            color='white', weight='bold')
    
    # Right side - applications folder
    right_icon = FancyBboxPatch((6.0, 0.8), 1.2, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#27ae60', edgecolor='#229954', linewidth=2)
    ax.add_patch(right_icon)
    ax.text(6.6, 1.4, '📁', fontsize=16, ha='center', va='center')
    
    # Add arrow
    arrow_x = [2.5, 5.5]
    arrow_y = [1.4, 1.4]
    ax.annotate('', xy=(5.5, 1.4), xytext=(2.5, 1.4),
                arrowprops=dict(arrowstyle='->', lw=3, color='#e74c3c'))
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Save the background
    plt.tight_layout()
    plt.savefig('assets/installer_background.png', dpi=150, bbox_inches='tight', 
                facecolor='#f0f4f8', edgecolor='none')
    plt.close()
    
    print("✅ Installer background created: assets/installer_background.png")

if __name__ == "__main__":
    create_installer_background()
