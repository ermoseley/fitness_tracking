#!/usr/bin/env python3

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


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
        self.root.geometry("640x420")
        # Allow both horizontal and vertical resizing
        self.root.resizable(True, True)
        self.root.minsize(640, 420)

        self.project_dir = os.path.dirname(os.path.abspath(__file__))

        # For bundled apps, we need to find the correct data directory
        # Check if we're running from a bundled app
        if getattr(sys, 'frozen', False):
            print("DEBUG: Running in bundled mode")
            # Running in a bundle - use the fitness_tracking directory for data files
            # This keeps all data in one place with the app
            bundle_dir = os.path.dirname(sys.executable)
            # Go up to the fitness_tracking directory (Contents/MacOS -> Contents -> fitness_tracking)
            data_dir = os.path.join(bundle_dir, "..", "..")
            print(f"DEBUG: Data directory will be: {data_dir}")

            # Ensure the directory exists
            os.makedirs(data_dir, exist_ok=True)
            print(f"DEBUG: Created data directory: {data_dir}")

            # Check if CSV files exist in bundle directory, if not copy from Resources
            # For bundled apps, the CSV files are in the Resources directory
            bundle_dir = os.path.dirname(sys.executable)
            bundle_resources = os.path.join(bundle_dir, "..", "Resources")
            bundle_csv = os.path.join(bundle_resources, "weights.csv")
            bundle_lbm = os.path.join(bundle_resources, "lbm.csv")
            print(f"DEBUG: Bundle CSV path: {bundle_csv}")
            print(f"DEBUG: Bundle LBM path: {bundle_lbm}")

            bundle_csv_dest = os.path.join(data_dir, "weights.csv")
            bundle_lbm_dest = os.path.join(data_dir, "lbm.csv")

            # Copy initial data files if they don't exist
            if not os.path.exists(bundle_csv_dest) and os.path.exists(bundle_csv):
                import shutil
                shutil.copy2(bundle_csv, bundle_csv_dest)
                print(f"DEBUG: Copied initial weights.csv to {bundle_csv_dest}")
            else:
                print(f"DEBUG: Bundle CSV already exists or source not found: {bundle_csv_dest}")

            if not os.path.exists(bundle_lbm_dest) and os.path.exists(bundle_lbm):
                import shutil
                shutil.copy2(bundle_lbm, bundle_lbm_dest)
                print(f"DEBUG: Copied initial lbm.csv to {bundle_lbm_dest}")
            else:
                print(f"DEBUG: Bundle LBM already exists or source not found: {bundle_lbm_dest}")
        else:
            print("DEBUG: Running in development mode")
            # Running from source
            data_dir = self.project_dir

        default_csv = os.path.join(data_dir, "weights.csv")
        default_lbm = os.path.join(data_dir, "lbm.csv")

        # Store data directory for use in other methods
        self.data_dir = data_dir

        # Variables
        self.var_csv = tk.StringVar(value=default_csv)
        self.var_lbm = tk.StringVar(value=default_lbm)
        self.var_start = tk.StringVar(value="")
        self.var_end = tk.StringVar(value="")
        self.var_kalman_mode = tk.StringVar(value="smoother")
        self.var_show_weight = tk.BooleanVar(value=True)
        self.var_show_kalman = tk.BooleanVar(value=True)
        self.var_no_display = tk.BooleanVar(value=False)

        # Add-entry variables (defaults)
        today_iso = date.today().isoformat()
        self.var_add_weight_date = tk.StringVar(value=today_iso)
        self.var_add_weight_value = tk.StringVar(value="")
        self.var_add_lbm_date = tk.StringVar(value=today_iso)
        self.var_add_lbm_value = tk.StringVar(value="")

        # Layout
        pad = {"padx": 8, "pady": 6}

        row = 0
        tk.Label(root, text="Weights CSV:").grid(row=row, column=0, sticky="e", **pad)
        tk.Entry(root, textvariable=self.var_csv, width=52).grid(row=row, column=1, sticky="we", **pad)
        tk.Button(root, text="Browse", command=self._browse_csv).grid(row=row, column=2, **pad)

        row += 1
        tk.Label(root, text="LBM CSV (optional):").grid(row=row, column=0, sticky="e", **pad)
        tk.Entry(root, textvariable=self.var_lbm, width=52).grid(row=row, column=1, sticky="we", **pad)
        tk.Button(root, text="Browse", command=self._browse_lbm).grid(row=row, column=2, **pad)

        row += 1
        tk.Label(root, text="Start date (YYYY-MM-DD):").grid(row=row, column=0, sticky="e", **pad)
        tk.Entry(root, textvariable=self.var_start, width=20).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Label(root, text="End date (YYYY-MM-DD):").grid(row=row, column=0, sticky="e", **pad)
        tk.Entry(root, textvariable=self.var_end, width=20).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Label(root, text="Kalman mode:").grid(row=row, column=0, sticky="e", **pad)
        tk.OptionMenu(root, self.var_kalman_mode, "smoother", "filter").grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Checkbutton(root, text="Generate Weight Plot", variable=self.var_show_weight).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Checkbutton(root, text="Generate Kalman + Body Fat plots", variable=self.var_show_kalman).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Checkbutton(root, text="Run headless (no display)", variable=self.var_no_display).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        tk.Button(root, text="Run", command=self._on_run, width=16).grid(row=row, column=1, sticky="w", **pad)
        tk.Button(root, text="Open weight plot", command=self._open_weight_plot).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        tk.Button(root, text="Open body fat plot", command=self._open_bodyfat_plot).grid(row=row, column=2, sticky="w", **pad)

        # Add entries section
        row += 1
        tk.Label(root, text="Add Weight Entry:").grid(row=row, column=0, sticky="e", **pad)
        weight_frame = tk.Frame(root)
        weight_frame.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        tk.Label(weight_frame, text="Date (YYYY-MM-DD):").grid(row=0, column=0, sticky="e", padx=(0, 6))
        tk.Entry(weight_frame, textvariable=self.var_add_weight_date, width=12).grid(row=0, column=1, sticky="w", padx=(0, 12))
        tk.Label(weight_frame, text="Weight (lb):").grid(row=0, column=2, sticky="e", padx=(0, 6))
        tk.Entry(weight_frame, textvariable=self.var_add_weight_value, width=10).grid(row=0, column=3, sticky="w", padx=(0, 12))
        tk.Button(weight_frame, text="Add to weights.csv", command=self._on_add_weight).grid(row=0, column=4, sticky="w")

        row += 1
        tk.Label(root, text="Add LBM Entry:").grid(row=row, column=0, sticky="e", **pad)
        lbm_frame = tk.Frame(root)
        lbm_frame.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        tk.Label(lbm_frame, text="Date (YYYY-MM-DD):").grid(row=0, column=0, sticky="e", padx=(0, 6))
        tk.Entry(lbm_frame, textvariable=self.var_add_lbm_date, width=12).grid(row=0, column=1, sticky="w", padx=(0, 12))
        tk.Label(lbm_frame, text="LBM (lb):").grid(row=0, column=2, sticky="e", padx=(0, 6))
        tk.Entry(lbm_frame, textvariable=self.var_add_lbm_value, width=10).grid(row=0, column=3, sticky="w", padx=(0, 12))
        tk.Button(lbm_frame, text="Add to LBM CSV", command=self._on_add_lbm).grid(row=0, column=4, sticky="w")

        row += 1
        self.output = scrolledtext.ScrolledText(root, height=14, width=80, wrap="word")
        self.output.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=8, pady=(0, 8))
        # Make the main content column and the output row expand with the window
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(row, weight=1)

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(title="Select weights CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.var_csv.set(path)

    def _browse_lbm(self) -> None:
        path = filedialog.askopenfilename(title="Select LBM CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.var_lbm.set(path)

    def _append_output(self, text: str) -> None:
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)

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

    def _get_plot_path(self, filename: str) -> str:
        """Get the correct path for plot files based on whether we're running bundled or not"""
        if getattr(sys, 'frozen', False):
            # Running in bundle - plots are in the fitness_tracking directory
            return os.path.join(self.data_dir, filename)
        else:
            # Running from source - plots are in project directory
            return os.path.join(self.project_dir, filename)

    def _open_weight_plot(self) -> None:
        """Open the weight trend plot"""
        path = self._get_plot_path("weight_trend.png")
        self._open_file(path)

    def _open_bodyfat_plot(self) -> None:
        """Open the body fat trend plot"""
        path = self._get_plot_path("bodyfat_trend.png")
        self._open_file(path)

    def _build_args(self) -> list[str]:
        # For bundled apps, we can import and run weight_tracker directly
        # This avoids subprocess issues with bundled executables
        if getattr(sys, 'frozen', False):
            # Running in a bundle - we'll import and run the module directly
            return []
        else:
            # Running from source - use system Python
            python_exe = sys.executable or "python3"
            script_path = os.path.join(self.project_dir, "weight_tracker.py")
            args = [python_exe, script_path]
            # CSVs
            if self.var_csv.get().strip():
                args += ["--csv", self.var_csv.get().strip()]
            if self.var_lbm.get().strip():
                args += ["--lbm-csv", self.var_lbm.get().strip()]
            # Dates
            if self.var_start.get().strip():
                args += ["--start", self.var_start.get().strip()]
            if self.var_end.get().strip():
                args += ["--end", self.var_end.get().strip()]
            # Mode and toggles
            args += ["--kalman-mode", self.var_kalman_mode.get().strip()]
            if not self.var_show_weight.get():
                args += ["--no-plot"]
            if not self.var_show_kalman.get():
                args += ["--no-kalman-plot"]
            if self.var_no_display.get():
                args += ["--no-display"]
            return args

    def _on_run(self) -> None:
        if getattr(sys, 'frozen', False):
            # Running in a bundle - import and run weight_tracker directly
            self._append_output("Running weight tracker (bundled mode)...")
            self._run_bundled_weight_tracker()
        else:
            # Running from source - use subprocess
            args = self._build_args()
            self._append_output("Running: " + " ".join(args))
            self._run_subprocess_weight_tracker(args)

    def _run_bundled_weight_tracker(self) -> None:
        """Run weight_tracker directly when in bundled mode"""
        def worker() -> None:
            try:
                # Import the weight_tracker module
                import weight_tracker

                # Create a mock argv for the module
                original_argv = sys.argv[:]
                sys.argv = ['weight_tracker.py']

                # Add arguments based on GUI settings
                if self.var_csv.get().strip():
                    sys.argv += ["--csv", self.var_csv.get().strip()]
                if self.var_lbm.get().strip():
                    sys.argv += ["--lbm-csv", self.var_lbm.get().strip()]
                if self.var_start.get().strip():
                    sys.argv += ["--start", self.var_start.get().strip()]
                if self.var_end.get().strip():
                    sys.argv += ["--end", self.var_end.get().strip()]
                sys.argv += ["--kalman-mode", self.var_kalman_mode.get().strip()]

                # Set output directory for plots to the fitness_tracking directory
                if getattr(sys, 'frozen', False):
                    output_dir = self.data_dir
                    os.makedirs(output_dir, exist_ok=True)
                    sys.argv += ["--output", os.path.join(output_dir, "weight_trend.png")]

                if not self.var_show_weight.get():
                    sys.argv += ["--no-plot"]
                if not self.var_show_kalman.get():
                    sys.argv += ["--no-kalman-plot"]
                
                # Allow user to choose display mode
                if self.var_no_display.get():
                    sys.argv += ["--no-display"]
                    print("DEBUG: Running in headless mode (user choice)")
                else:
                    print("DEBUG: Running with interactive plots enabled")

                # Set matplotlib backend
                import matplotlib
                if self.var_no_display.get():
                    # Headless mode - use Agg backend
                    matplotlib.use("Agg")
                    print("DEBUG: Using Agg backend (headless mode)")
                else:
                    # Interactive mode - try to use TkAgg for better integration
                    try:
                        # For bundled apps, we need to ensure matplotlib can find the Tkinter backend
                        if getattr(sys, 'frozen', False):
                            # Set environment variable to help matplotlib find Tkinter
                            os.environ['TK_SILENCE_DEPRECATION'] = '1'
                            
                        matplotlib.use("TkAgg")
                        print("DEBUG: Set TkAgg backend for interactive plotting")
                    except Exception as e:
                        print(f"DEBUG: TkAgg backend failed ({e}), falling back to Agg")
                        matplotlib.use("Agg")

                # Change working directory to the fitness_tracking directory for bundled mode
                original_cwd = os.getcwd()
                if getattr(sys, 'frozen', False):
                    os.chdir(self.data_dir)

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

                # Restore original argv and cwd after weight_tracker execution
                sys.argv = original_argv
                if getattr(sys, 'frozen', False):
                    os.chdir(original_cwd)

            except Exception as e:
                self._append_output(f"Error: {e}")
                import traceback
                self._append_output(f"Traceback: {traceback.format_exc()}")

        threading.Thread(target=worker, daemon=True).start()

    def _run_subprocess_weight_tracker(self, args: list[str]) -> None:
        """Run weight_tracker using subprocess (for development mode)"""
        def worker() -> None:
            try:
                env = os.environ.copy()
                # Ensure matplotlib uses a GUI backend when not headless; Agg when headless
                if self.var_no_display.get():
                    env["MPLBACKEND"] = "Agg"
                else:
                    env["MPLBACKEND"] = "MacOSX"

                proc = subprocess.Popen(args, cwd=self.project_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._append_output(line.rstrip())
                code = proc.wait()
                self._append_output(f"Process exited with code {code}")
                if code == 0:
                    self._append_output("Done. You can click the buttons to open the generated plots.")
            except Exception as e:
                self._append_output("Error: " + str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _validate_iso_date(self, s: str) -> bool:
        try:
            datetime.fromisoformat(s)
            return True
        except Exception:
            return False

    def _on_add_weight(self) -> None:
        d = self.var_add_weight_date.get().strip()
        w_str = self.var_add_weight_value.get().strip()
        if not self._validate_iso_date(d):
            messagebox.showerror("Invalid date", "Please enter a valid date in YYYY-MM-DD format for weight entry.")
            return
        try:
            float(w_str)
        except Exception:
            messagebox.showerror("Invalid weight", "Please enter a numeric weight value.")
            return

        args = [sys.executable or "python3", os.path.join(self.project_dir, "weight_tracker.py"),
                "--csv", self.var_csv.get().strip(), "--add", f"{d}:{w_str}",
                "--no-plot", "--no-kalman-plot", "--no-display"]

        self._append_output("Appending weight entry via CLI: " + " ".join(args))

        def worker() -> None:
            try:
                proc = subprocess.Popen(args, cwd=self.project_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._append_output(line.rstrip())
                code = proc.wait()
                if code == 0:
                    self._append_output("Weight entry added.")
                else:
                    self._append_output(f"Failed to add weight entry (exit {code}).")
            except Exception as e:
                self._append_output("Error adding weight entry: " + str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_add_lbm(self) -> None:
        d = self.var_add_lbm_date.get().strip()
        v_str = self.var_add_lbm_value.get().strip()
        if not self._validate_iso_date(d):
            messagebox.showerror("Invalid date", "Please enter a valid date in YYYY-MM-DD format for LBM entry.")
            return
        try:
            float(v_str)
        except Exception:
            messagebox.showerror("Invalid LBM", "Please enter a numeric LBM value.")
            return

        target = self.var_lbm.get().strip() or os.path.join(self.project_dir, "lbm.csv")
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
                    f.write("date,lbm\n")
                f.write(f"{d},{v_str}\n")
            self._append_output(f"LBM entry added to {target}: {d},{v_str}")
        except Exception as e:
            messagebox.showerror("LBM append failed", str(e))
            self._append_output("LBM append failed: " + str(e))


def main() -> None:
    root = tk.Tk()
    WeightTrackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


