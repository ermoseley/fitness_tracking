#!/usr/bin/env python3
"""
Main entry point for WeightTracker
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add the current directory to the path so we can import gui
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import WeightTrackerGUI

def main():
    """Main function to launch the Weight Tracker GUI"""
    try:
        # Create the main window
        root = tk.Tk()
        
        # Set the app icon if available
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon_128.png")
        if os.path.exists(icon_path):
            try:
                icon_image = tk.PhotoImage(file=icon_path)
                root.iconphoto(True, icon_image)
            except Exception:
                pass  # Icon loading is optional
        
        # Create and run the GUI
        app = WeightTrackerGUI(root)
        
        # Start the main event loop
        root.mainloop()
        
    except Exception as e:
        # Show error dialog if something goes wrong
        error_root = tk.Tk()
        error_root.withdraw()
        
        messagebox.showerror("Error", f"Failed to launch Weight Tracker:\n{str(e)}")
        error_root.destroy()
        sys.exit(1)

if __name__ == "__main__":
    main()
