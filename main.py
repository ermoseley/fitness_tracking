#!/usr/bin/env python3
"""
Main entry point for Weight Tracker macOS App
This script launches the GUI application when the app bundle is double-clicked.
"""

import os
import sys
import tkinter as tk
from pathlib import Path

# Add the current directory to Python path so we can import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from gui import WeightTrackerGUI
except ImportError as e:
    # If we can't import, show an error dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    error_dialog = tk.Toplevel(root)
    error_dialog.title("Import Error")
    error_dialog.geometry("400x200")
    error_dialog.resizable(False, False)
    
    # Center the dialog
    error_dialog.transient(root)
    error_dialog.grab_set()
    
    tk.Label(error_dialog, text="Failed to import required modules", font=("Arial", 14, "bold")).pack(pady=20)
    tk.Label(error_dialog, text=f"Error: {str(e)}", wraplength=350).pack(pady=10)
    tk.Label(error_dialog, text="Please ensure all dependencies are installed", wraplength=350).pack(pady=10)
    
    tk.Button(error_dialog, text="OK", command=lambda: [error_dialog.destroy(), root.destroy()]).pack(pady=20)
    
    root.mainloop()
    sys.exit(1)

def main():
    """Main function to launch the Weight Tracker GUI"""
    try:
        # Create the main window
        root = tk.Tk()
        
        # Set the app icon if available
        icon_path = current_dir / "assets" / "icon_128.png"
        if icon_path.exists():
            try:
                icon_image = tk.PhotoImage(file=str(icon_path))
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
        
        error_dialog = tk.Toplevel(error_root)
        error_dialog.title("Application Error")
        error_dialog.geometry("400x200")
        error_dialog.resizable(False, False)
        
        error_dialog.transient(error_root)
        error_dialog.grab_set()
        
        tk.Label(error_dialog, text="An error occurred while running the application", font=("Arial", 14, "bold")).pack(pady=20)
        tk.Label(error_dialog, text=f"Error: {str(e)}", wraplength=350).pack(pady=10)
        
        tk.Button(error_dialog, text="OK", command=lambda: [error_dialog.destroy(), error_root.destroy()]).pack(pady=20)
        
        error_root.mainloop()
        sys.exit(1)

if __name__ == "__main__":
    main()
