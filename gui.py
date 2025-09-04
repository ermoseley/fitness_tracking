#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


@dataclass
class GuiState:
    csv_path: str
    lbm_csv: str
    start_date: str
    end_date: str
    no_plot: bool
    no_kalman: bool
    kalman_mode: str  # "smoother" or "filter"


class WeightTrackerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Weight Tracker")
        self.root.geometry("1080x880")
        # Allow both horizontal and vertical resizing
        self.root.resizable(True, True)
        self.root.minsize(640, 420)
        
        # Fix for macOS button responsiveness issues
        self.root.update_idletasks()
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(lambda: self.root.attributes('-topmost', False))
        
        # Note: Avoid global focus/click bindings which can steal focus from inputs on macOS
        
        # Disable aggressive periodic refresh that causes responsiveness issues
        # self._refresh_timer_id = None
        # self._schedule_gui_refresh()

        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.project_dir, "data")
        default_csv = ""
        default_lbm = ""

        # Variables
        self.var_csv = tk.StringVar(value=default_csv)
        self.var_fat_mass = tk.StringVar(value=default_lbm)
        self.var_start = tk.StringVar(value="")
        self.var_end = tk.StringVar(value="")
        self.var_kalman_mode = tk.StringVar(value="smoother")
        self.var_confidence_interval = tk.StringVar(value="95%")
        self.var_show_weight = tk.BooleanVar(value=True)
        self.var_show_weight_plot = tk.BooleanVar(value=True)
        self.var_show_bodyfat_plot = tk.BooleanVar(value=True)
        self.var_show_bmi_plot = tk.BooleanVar(value=False)
        self.var_show_ffmi_plot = tk.BooleanVar(value=False)
        self.var_show_lbm_plot = tk.BooleanVar(value=False)
        self.var_show_fatmass_plot = tk.BooleanVar(value=False)
        self.var_show_residuals_plot = tk.BooleanVar(value=False)
        self.var_aggregation_hours = tk.StringVar(value="3")
        self.var_no_display = tk.BooleanVar(value=False)

        # Add-entry variables (defaults)
        today_iso = date.today().isoformat()
        self.var_add_weight_date = tk.StringVar(value=today_iso)
        self.var_add_weight_value = tk.StringVar(value="")
        self.var_add_fat_mass_date = tk.StringVar(value=today_iso)
        self.var_add_fat_mass_value = tk.StringVar(value="")
        self.var_add_bf_date = tk.StringVar(value=today_iso)
        self.var_add_bf_value = tk.StringVar(value="")
        self.var_height_value = tk.StringVar(value="")
        self.var_height_unit = tk.StringVar(value="inches")

        # Layout
        pad = {"padx": 8, "pady": 6}

        row = 0
        tk.Label(root, text="Upload weights CSV:").grid(row=row, column=0, sticky="e", **pad)
        tk.Entry(root, textvariable=self.var_csv, width=52).grid(row=row, column=1, sticky="we", **pad)
        tk.Button(root, text="Browse", command=self._browse_csv).grid(row=row, column=2, **pad)

        row += 1
        tk.Label(root, text="Upload Fat Mass CSV (optional):").grid(row=row, column=0, sticky="e", **pad)
        tk.Entry(root, textvariable=self.var_fat_mass, width=52).grid(row=row, column=1, sticky="we", **pad)
        tk.Button(root, text="Browse", command=self._browse_fat_mass).grid(row=row, column=2, **pad)

        row += 1
        tk.Label(root, text="Start date (YYYY-MM-DD[THH:MM:SS]):").grid(row=row, column=0, sticky="e", **pad)
        tk.Entry(root, textvariable=self.var_start, width=20).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Label(root, text="End date (YYYY-MM-DD[THH:MM:SS]):").grid(row=row, column=0, sticky="e", **pad)
        tk.Entry(root, textvariable=self.var_end, width=20).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Label(root, text="Kalman mode:").grid(row=row, column=0, sticky="e", **pad)
        tk.OptionMenu(root, self.var_kalman_mode, "smoother", "filter").grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Label(root, text="Confidence interval:").grid(row=row, column=0, sticky="e", **pad)
        tk.OptionMenu(root, self.var_confidence_interval, "95%", "1σ").grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Label(root, text="Aggregation window (hours):").grid(row=row, column=0, sticky="e", **pad)
        tk.OptionMenu(root, self.var_aggregation_hours, "3", "6", "12", "24", "0 (off)").grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Label(root, text="Generate plots:").grid(row=row, column=0, sticky="e", **pad)
        plot_frame = tk.Frame(root)
        plot_frame.grid(row=row, column=1, sticky="w", **pad)
        tk.Checkbutton(plot_frame, text="Weight trend", variable=self.var_show_weight_plot).grid(row=0, column=0, sticky="w", padx=(0, 12))
        tk.Checkbutton(plot_frame, text="Body fat %", variable=self.var_show_bodyfat_plot).grid(row=0, column=1, sticky="w", padx=(0, 12))
        tk.Checkbutton(plot_frame, text="BMI", variable=self.var_show_bmi_plot).grid(row=0, column=2, sticky="w", padx=(0, 12))
        tk.Checkbutton(plot_frame, text="FFMI", variable=self.var_show_ffmi_plot).grid(row=0, column=3, sticky="w", padx=(0, 12))
        tk.Checkbutton(plot_frame, text="LBM", variable=self.var_show_lbm_plot).grid(row=0, column=4, sticky="w", padx=(0, 12))
        tk.Checkbutton(plot_frame, text="Fat mass (lb)", variable=self.var_show_fatmass_plot).grid(row=0, column=5, sticky="w")

        row += 1
        tk.Checkbutton(root, text="Save plots as PNG files", variable=self.var_show_weight).grid(row=row, column=1, sticky="w", **pad)
        
        row += 1
        tk.Checkbutton(root, text="Run headless (no python GUI for plots)", variable=self.var_no_display).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Button(root, text="Run", command=self._on_run, width=16).grid(row=row, column=1, sticky="w", **pad)
        tk.Button(root, text="Open weight plot", command=lambda: self._open_file(os.path.join(self.project_dir, "weight_trend.png"))).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Button(root, text="Open body fat plot", command=lambda: self._open_file(os.path.join(self.project_dir, "bodyfat_trend.png"))).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Button(root, text="Open BMI plot", command=lambda: self._open_file(os.path.join(self.project_dir, "bmi_trend.png"))).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Button(root, text="Open FFMI plot", command=lambda: self._open_file(os.path.join(self.project_dir, "ffmi_trend.png"))).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Button(root, text="Open residuals plot", command=lambda: self._open_file(os.path.join(self.project_dir, "residuals_histogram.png"))).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Button(root, text="Open LBM plot", command=lambda: self._open_file(os.path.join(self.project_dir, "lbm_trend.png"))).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Button(root, text="Open Fat mass plot", command=lambda: self._open_file(os.path.join(self.project_dir, "fatmass_trend.png"))).grid(row=row, column=2, sticky="w", **pad)

        # Add entries section
        row += 1
        tk.Label(root, text="Add Weight Entry:").grid(row=row, column=0, sticky="e", **pad)
        weight_frame = tk.Frame(root)
        weight_frame.grid(row=row, column=1, sticky="we", **pad)
        # Configure weight_frame columns for consistent alignment
        weight_frame.grid_columnconfigure(0, weight=0)  # DateTime label
        weight_frame.grid_columnconfigure(1, weight=0)  # DateTime entry
        weight_frame.grid_columnconfigure(2, weight=0)  # Weight label
        weight_frame.grid_columnconfigure(3, weight=0)  # Weight entry
        tk.Label(weight_frame, text="DateTime (YYYY-MM-DD[THH:MM:SS]):").grid(row=0, column=0, sticky="e", padx=(0, 6))
        tk.Entry(weight_frame, textvariable=self.var_add_weight_date, width=12).grid(row=0, column=1, sticky="w", padx=(0, 12))
        tk.Label(weight_frame, text="Weight (lb):", width=12, anchor="e").grid(row=0, column=2, sticky="e", padx=(0, 6))
        tk.Entry(weight_frame, textvariable=self.var_add_weight_value, width=10).grid(row=0, column=3, sticky="w", padx=(0, 12))
        tk.Button(root, text="Add to weights.csv", command=self._on_add_weight).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Label(root, text="Add Fat Mass Entry:").grid(row=row, column=0, sticky="e", **pad)
        fat_mass_frame = tk.Frame(root)
        fat_mass_frame.grid(row=row, column=1, sticky="we", **pad)
        # Configure fat_mass_frame columns to match weight_frame
        fat_mass_frame.grid_columnconfigure(0, weight=0)  # DateTime label
        fat_mass_frame.grid_columnconfigure(1, weight=0)  # DateTime entry
        fat_mass_frame.grid_columnconfigure(2, weight=0)  # Fat Mass label
        fat_mass_frame.grid_columnconfigure(3, weight=0)  # Fat Mass entry
        tk.Label(fat_mass_frame, text="DateTime (YYYY-MM-DD[THH:MM:SS]):").grid(row=0, column=0, sticky="e", padx=(0, 6))
        tk.Entry(fat_mass_frame, textvariable=self.var_add_fat_mass_date, width=12).grid(row=0, column=1, sticky="w", padx=(0, 12))
        tk.Label(fat_mass_frame, text="Fat Mass (lb):", width=12, anchor="e").grid(row=0, column=2, sticky="e", padx=(0, 6))
        tk.Entry(fat_mass_frame, textvariable=self.var_add_fat_mass_value, width=10).grid(row=0, column=3, sticky="w", padx=(0, 12))
        tk.Button(root, text="Add to Fat Mass CSV", command=self._on_add_fat_mass).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Label(root, text="Add Body Fat %:").grid(row=row, column=0, sticky="e", **pad)
        bf_frame = tk.Frame(root)
        bf_frame.grid(row=row, column=1, sticky="we", **pad)
        # Configure bf_frame columns to match other frames
        bf_frame.grid_columnconfigure(0, weight=0)  # DateTime label
        bf_frame.grid_columnconfigure(1, weight=0)  # DateTime entry
        bf_frame.grid_columnconfigure(2, weight=0)  # BF label
        bf_frame.grid_columnconfigure(3, weight=0)  # BF entry
        tk.Label(bf_frame, text="DateTime (YYYY-MM-DD[THH:MM:SS]):").grid(row=0, column=0, sticky="e", padx=(0, 6))
        tk.Entry(bf_frame, textvariable=self.var_add_bf_date, width=12).grid(row=0, column=1, sticky="w", padx=(0, 12))
        tk.Label(bf_frame, text="Body Fat (%):", width=12, anchor="e").grid(row=0, column=2, sticky="e", padx=(0, 6))
        tk.Entry(bf_frame, textvariable=self.var_add_bf_value, width=10).grid(row=0, column=3, sticky="w", padx=(0, 12))
        tk.Button(root, text="Add to LBM CSV", command=self._on_add_bf).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Label(root, text="Height:").grid(row=row, column=0, sticky="e", **pad)
        height_frame = tk.Frame(root)
        height_frame.grid(row=row, column=1, sticky="we", **pad)
        # Configure height_frame columns
        height_frame.grid_columnconfigure(0, weight=0)  # Height label
        height_frame.grid_columnconfigure(1, weight=0)  # Height entry
        height_frame.grid_columnconfigure(2, weight=0)  # Unit selector
        tk.Label(height_frame, text="Height:", width=12, anchor="e").grid(row=0, column=0, sticky="e", padx=(0, 6))
        tk.Entry(height_frame, textvariable=self.var_height_value, width=10).grid(row=0, column=1, sticky="w", padx=(0, 6))
        tk.OptionMenu(height_frame, self.var_height_unit, "inches", "cm").grid(row=0, column=2, sticky="w", padx=(6, 0))
        tk.Button(root, text="Update Height", command=self._on_update_height).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        self.output = scrolledtext.ScrolledText(root, height=14, width=80, wrap="word")
        self.output.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=8, pady=(0, 8))
        # Make the main content column and the output row expand with the window
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(row, weight=1)
        
        # Load current height value after GUI is set up
        self._load_height_value()

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(title="Select weights CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.var_csv.set(path)
        # Simple focus restoration
        self.root.focus_force()
        self.root.lift()

    def _browse_fat_mass(self) -> None:
        path = filedialog.askopenfilename(title="Select Fat Mass CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.var_fat_mass.set(path)
        # Simple focus restoration
        self.root.focus_force()
        self.root.lift()

    def _append_output(self, text: str) -> None:
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)
        # Force GUI update to prevent button freezing
        self.root.update_idletasks()
    
    def _append_output_safe(self, text: str) -> None:
        """Thread-safe append to output text widget."""
        self.root.after(0, lambda: self._append_output(text))
    
    def _refresh_gui(self) -> None:
        """Refresh GUI to prevent button freezing issues"""
        self.root.update_idletasks()
        # Less aggressive update
        self.root.after(10, lambda: None)
    
    def _on_focus_in(self, event) -> None:
        """Handle window focus events to ensure responsiveness"""
        self.root.update_idletasks()
    
    def _on_click(self, event) -> None:
        """Handle click events to ensure window stays active"""
        self.root.focus_force()
        self.root.update_idletasks()
    
    def _schedule_gui_refresh(self) -> None:
        """Schedule periodic GUI refresh to prevent freezing"""
        self.root.update_idletasks()
        # Schedule next refresh in 500ms (less aggressive)
        self._refresh_timer_id = self.root.after(500, self._schedule_gui_refresh)
    
    def _restart_refresh_timer(self) -> None:
        """Restart the refresh timer after file dialogs"""
        self._schedule_gui_refresh()

    def _open_file(self, path: str) -> None:
        if not os.path.exists(path):
            messagebox.showinfo("Not found", f"File not found: {path}")
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _build_args(self) -> list[str]:
        # Check if we're running in a bundled app
        if getattr(sys, 'frozen', False):
            # Running in a bundle - we need to import and run weight_tracker directly
            return []
        else:
            # Running from source - use system Python
            args: list[str] = ["python3", os.path.join(self.project_dir, "weight_tracker.py")]
            # Ensure data dir exists
            os.makedirs(self.data_dir, exist_ok=True)
            # CSVs (upload semantics)
            weights_target = os.path.join(self.data_dir, "weights.csv")
            fat_mass_target = os.path.join(self.data_dir, "fat_mass.csv")
            src_weights = self.var_csv.get().strip()
            src_fat_mass = self.var_fat_mass.get().strip()
            if src_weights:
                try:
                    if os.path.abspath(src_weights) != os.path.abspath(weights_target):
                        shutil.copy2(src_weights, weights_target)
                except Exception as e:
                    messagebox.showerror("Upload failed", f"Failed to upload weights CSV: {e}")
                args += ["--csv", weights_target]
            # If not provided, rely on default in weight_tracker (data/weights.csv)
            if src_fat_mass:
                try:
                    if os.path.abspath(src_fat_mass) != os.path.abspath(fat_mass_target):
                        shutil.copy2(src_fat_mass, fat_mass_target)
                except Exception as e:
                    messagebox.showerror("Upload failed", f"Failed to upload Fat Mass CSV: {e}")
                args += ["--fat-mass-csv", fat_mass_target]
            # Dates
            if self.var_start.get().strip():
                args += ["--start", self.var_start.get().strip()]
            if self.var_end.get().strip():
                args += ["--end", self.var_end.get().strip()]
            # Mode and toggles
            args += ["--kalman-mode", self.var_kalman_mode.get().strip()]
            args += ["--confidence-interval", self.var_confidence_interval.get().strip()]
            # Aggregation window
            agg = self.var_aggregation_hours.get().strip()
            if agg == "0 (off)":
                args += ["--aggregation-hours", "0"]
            elif agg:
                args += ["--aggregation-hours", agg]
            # Plot controls
            if not self.var_show_weight_plot.get():
                args += ["--no-kalman-plot", "--no-plot"]  # Disable both weight trend plots
            if not self.var_show_bodyfat_plot.get():
                args += ["--no-bodyfat-plot"]
            if not self.var_show_bmi_plot.get():
                args += ["--no-bmi-plot"]
            if not self.var_show_ffmi_plot.get():
                args += ["--no-ffmi-plot"]
            if not self.var_show_lbm_plot.get():
                args += ["--no-lbm-plot"]
            if not self.var_show_fatmass_plot.get():
                args += ["--no-fatmass-plot"]
            # LBM and Fat mass always enabled for now (no checkboxes)
            if self.var_show_residuals_plot.get():
                args += ["--residuals-histogram"]
            if not self.var_show_weight.get():
                args += ["--no-plot"]
            if self.var_no_display.get():
                args += ["--no-display"]
            return args

    def _on_run(self) -> None:
        args = self._build_args()
        
        if getattr(sys, 'frozen', False):
            # Running in a bundle - import and run weight_tracker directly
            self._append_output("Running weight tracker (bundled mode)...")
            self._run_bundled_weight_tracker()
        else:
            # Running from source - use subprocess
            self._append_output("Running: " + " ".join(args))
            self._run_subprocess_weight_tracker(args)

    def _run_bundled_weight_tracker(self) -> None:
        """Run weight_tracker directly when in bundled mode"""
        try:
            # Import the weight_tracker module
            import weight_tracker
            
            # Create a mock argv for the module
            original_argv = sys.argv[:]
            sys.argv = ['weight_tracker.py']
            
            # Ensure data dir exists
            os.makedirs(self.data_dir, exist_ok=True)
            weights_target = os.path.join(self.data_dir, "weights.csv")
            fat_mass_target = os.path.join(self.data_dir, "fat_mass.csv")
            # Add arguments based on GUI settings (upload semantics)
            if self.var_csv.get().strip():
                src_weights = self.var_csv.get().strip()
                try:
                    if os.path.abspath(src_weights) != os.path.abspath(weights_target):
                        shutil.copy2(src_weights, weights_target)
                except Exception as e:
                    self._append_output(f"Upload weights CSV failed: {e}")
                sys.argv += ["--csv", weights_target]
            if self.var_fat_mass.get().strip():
                src_fat_mass = self.var_fat_mass.get().strip()
                try:
                    if os.path.abspath(src_fat_mass) != os.path.abspath(fat_mass_target):
                        shutil.copy2(src_fat_mass, fat_mass_target)
                except Exception as e:
                    self._append_output(f"Upload Fat Mass CSV failed: {e}")
                sys.argv += ["--fat-mass-csv", fat_mass_target]
            if self.var_start.get().strip():
                sys.argv += ["--start", self.var_start.get().strip()]
            if self.var_end.get().strip():
                sys.argv += ["--end", self.var_end.get().strip()]
            sys.argv += ["--kalman-mode", self.var_kalman_mode.get().strip()]
            sys.argv += ["--confidence-interval", self.var_confidence_interval.get().strip()]
            
            # Plot controls
            if not self.var_show_weight_plot.get():
                sys.argv += ["--no-kalman-plot", "--no-plot"]  # Disable both weight trend plots
            if not self.var_show_bodyfat_plot.get():
                sys.argv += ["--no-bodyfat-plot"]
            if not self.var_show_bmi_plot.get():
                sys.argv += ["--no-bmi-plot"]
            if not self.var_show_ffmi_plot.get():
                sys.argv += ["--no-ffmi-plot"]
            if not self.var_show_lbm_plot.get():
                sys.argv += ["--no-lbm-plot"]
            if not self.var_show_fatmass_plot.get():
                sys.argv += ["--no-fatmass-plot"]
            # LBM and Fat mass always enabled for now (no checkboxes)
            if self.var_show_residuals_plot.get():
                sys.argv += ["--residuals-histogram"]
            if not self.var_show_weight.get():
                sys.argv += ["--no-plot"]
            if self.var_no_display.get():
                sys.argv += ["--no-display"]
            
            # Set matplotlib backend
            import matplotlib
            if self.var_no_display.get():
                matplotlib.use("Agg")
            else:
                matplotlib.use("TkAgg")
            
            # Capture stdout to display in GUI
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    weight_tracker.main()
                    output = f.getvalue()
                    self._append_output(output)
                    self._append_output("Done. You can click the buttons to open the generated plots.")
                except SystemExit as e:
                    if e.code == 0:
                        self._append_output("Done. You can click the buttons to open the generated plots.")
                    else:
                        self._append_output(f"Weight tracker exited with code {e.code}")
                except Exception as e:
                    self._append_output(f"Error running weight tracker: {e}")
                    import traceback
                    self._append_output(f"Traceback: {traceback.format_exc()}")
            
            # Restore original argv
            sys.argv = original_argv
            
        except Exception as e:
            self._append_output(f"Error: {e}")
            import traceback
            self._append_output(f"Traceback: {traceback.format_exc()}")

    def _run_subprocess_weight_tracker(self, args: list[str]) -> None:
        """Run weight_tracker using subprocess (for development mode)"""
        def worker() -> None:
            try:
                env = os.environ.copy()
                # Ensure matplotlib uses a GUI backend when not headless; Agg when headless
                if self.var_no_display.get():
                    env["MPLBACKEND"] = "Agg"
                else:
                    # Prefer native macOS backend if available; fallback handled by matplotlib
                    env["MPLBACKEND"] = "MacOSX"
                proc = subprocess.Popen(args, cwd=self.project_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._append_output_safe(line.rstrip())
                code = proc.wait()
                self._append_output_safe(f"Process exited with code {code}")
                if code == 0:
                    self._append_output_safe("Done. You can click the buttons to open the generated plots.")
                # Final GUI refresh
                self.root.after(1, self._refresh_gui)
            except Exception as e:
                self._append_output_safe("Error: " + str(e))
                self.root.after(1, self._refresh_gui)

        threading.Thread(target=worker, daemon=True).start()

    def _validate_datetime(self, s: str) -> bool:
        try:
            from weight_tracker import parse_datetime
            parse_datetime(s)
            return True
        except Exception:
            return False

    def _on_add_weight(self) -> None:
        d = self.var_add_weight_date.get().strip()
        w_str = self.var_add_weight_value.get().strip()
        if not self._validate_datetime(d):
            messagebox.showerror("Invalid datetime", "Please enter a valid datetime in YYYY-MM-DD[THH:MM:SS] format for weight entry.")
            return
        try:
            float(w_str)
        except Exception:
            messagebox.showerror("Invalid weight", "Please enter a numeric weight value.")
            return

        # Ensure data dir exists and target path chosen
        os.makedirs(self.data_dir, exist_ok=True)
        weights_target = os.path.join(self.data_dir, "weights.csv")

        args = [sys.executable or "python3", os.path.join(self.project_dir, "weight_tracker.py"),
                "--csv", weights_target, "--add", f"{d}:{w_str}",
                "--no-plot", "--no-kalman-plot", "--no-display"]

        self._append_output_safe("Appending weight entry via CLI: " + " ".join(args))

        def worker() -> None:
            try:
                proc = subprocess.Popen(args, cwd=self.project_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._append_output_safe(line.rstrip())
                code = proc.wait()
                if code == 0:
                    self._append_output_safe("Weight entry added.")
                else:
                    self._append_output_safe(f"Failed to add weight entry (exit {code}).")
                # Final GUI refresh
                pass
            except Exception as e:
                self._append_output_safe("Error adding weight entry: " + str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_add_fat_mass(self) -> None:
        d = self.var_add_fat_mass_date.get().strip()
        v_str = self.var_add_fat_mass_value.get().strip()
        if not self._validate_datetime(d):
            messagebox.showerror("Invalid datetime", "Please enter a valid datetime in YYYY-MM-DD[THH:MM:SS] format for Fat Mass entry.")
            return
        try:
            float(v_str)
        except Exception:
            messagebox.showerror("Invalid Fat Mass", "Please enter a numeric Fat Mass value.")
            return

        os.makedirs(self.data_dir, exist_ok=True)
        target = os.path.join(self.data_dir, "fat_mass.csv")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            need_header = not os.path.exists(target) or os.path.getsize(target) == 0
            need_leading_newline = False
            if not need_header:
                try:
                    with open(target, "rb") as frb:
                        frb.seek(-1, os.SEEK_END)
                        last_byte = frb.read(1)
                        if last_byte not in (b"\n", b"\r"):
                            need_leading_newline = True
                except OSError:
                    pass
            with open(target, "a", newline="") as f:
                if need_leading_newline:
                    f.write("\n")
                if need_header:
                    f.write("date,fat_mass\n")
                f.write(f"{d},{v_str}\n")
            self._append_output_safe(f"Fat Mass entry added to {target}: {d},{v_str}")
        except Exception as e:
            messagebox.showerror("Fat Mass append failed", str(e))
            self._append_output_safe("Fat Mass append failed: " + str(e))

    def _on_add_bf(self) -> None:
        d = self.var_add_bf_date.get().strip()
        bf_str = self.var_add_bf_value.get().strip()
        if not self._validate_datetime(d):
            messagebox.showerror("Invalid datetime", "Please enter a valid datetime in YYYY-MM-DD[THH:MM:SS] format for body fat entry.")
            return
        try:
            bf_percent = float(bf_str)
            if bf_percent < 0 or bf_percent > 100:
                messagebox.showerror("Invalid body fat", "Body fat percentage must be between 0 and 100.")
                return
        except Exception:
            messagebox.showerror("Invalid body fat", "Please enter a numeric body fat percentage.")
            return

        # Run Kalman calculation in background thread
        def worker() -> None:
            try:
                from weight_tracker import load_entries
                from kalman import run_kalman_smoother
                
                # Load current data
                weights_path = os.path.join(self.data_dir, "weights.csv")
                if not os.path.exists(weights_path):
                    self._append_output("Error: No weight data found. Please add weight entries first.")
                    return
                
                entries = load_entries(weights_path)
                if not entries:
                    self._append_output("Error: No weight data found. Please add weight entries first.")
                    return
                
                # Run Kalman filter to get current weight estimate
                kalman_states, kalman_dates = run_kalman_smoother(entries)
                if not kalman_states:
                    self._append_output("Error: Failed to get weight estimate from Kalman filter.")
                    return
                
                # Get the most recent weight estimate
                current_weight = kalman_states[-1].weight
                
                # Calculate LBM: LBM = weight * (1 - body_fat_percent/100)
                lbm = current_weight * (1.0 - bf_percent / 100.0)
                
                # Add to LBM CSV
                os.makedirs(self.data_dir, exist_ok=True)
                target = os.path.join(self.data_dir, "lbm.csv")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
                need_header = not os.path.exists(target) or os.path.getsize(target) == 0
                need_leading_newline = False
                if not need_header:
                    try:
                        size = os.path.getsize(target)
                        if size > 0:
                            with open(target, "rb") as frb:
                                try:
                                    frb.seek(-1, os.SEEK_END)
                                    last_byte = frb.read(1)
                                    if last_byte not in (b"\n", b"\r"):
                                        need_leading_newline = True
                                except OSError:
                                    pass
                    except OSError:
                        pass
                with open(target, "a", newline="") as f:
                    if need_leading_newline:
                        f.write("\n")
                    if need_header:
                        f.write("date,lbm\n")
                    f.write(f"{d},{lbm:.3f}\n")
                
                self._append_output_safe(f"Body fat {bf_percent}% converted to LBM {lbm:.3f} lb and added to {target}")
                
            except Exception as e:
                self._append_output_safe("Body fat conversion failed: " + str(e))
        
        self._append_output_safe("Converting body fat percentage to LBM...")
        threading.Thread(target=worker, daemon=True).start()

    def _load_height_value(self) -> None:
        """Load current height value from data/height.txt or set default"""
        height_file = os.path.join(self.data_dir, "height.txt")
        
        try:
            if os.path.exists(height_file):
                with open(height_file, 'r') as f:
                    height_inches = float(f.read().strip())
                # Display in inches by default
                self.var_height_value.set(f"{height_inches:.1f}")
                self.var_height_unit.set("inches")
            else:
                # Set default value (67 inches = 170 cm)
                self.var_height_value.set("67.0")
                self.var_height_unit.set("inches")
        except Exception:
            # If file exists but can't be read, use default
            self.var_height_value.set("67.0")
            self.var_height_unit.set("inches")

    def _on_update_height(self) -> None:
        """Update the persistent height value"""
        height_str = self.var_height_value.get().strip()
        unit = self.var_height_unit.get().strip()
        
        try:
            height_value = float(height_str)
            if height_value <= 0:
                messagebox.showerror("Invalid height", "Height must be greater than 0.")
                return
        except Exception:
            messagebox.showerror("Invalid height", "Please enter a numeric height value.")
            return

        # Convert to inches if needed
        if unit == "cm":
            height_inches = height_value * 0.393701  # cm to inches
        else:  # inches
            height_inches = height_value

        # Save to height.txt (single value, not time-series)
        os.makedirs(self.data_dir, exist_ok=True)
        target = os.path.join(self.data_dir, "height.txt")
        
        with open(target, "w") as f:
            f.write(f"{height_inches:.3f}")
        
        self._append_output(f"Height updated to {height_value} {unit} ({height_inches:.3f} inches)")



def main() -> None:
    root = tk.Tk()
    WeightTrackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


