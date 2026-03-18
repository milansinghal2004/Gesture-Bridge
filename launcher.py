import tkinter as tk
import subprocess
import os

# Define script execution functions
def run_script(script_path):
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")

# Callbacks
def detect_options():
    clear_frame()
    # Corrected relative paths to run regardless of exact clone path
    base_path = os.path.dirname(os.path.abspath(__file__))
    tk.Button(root, text="Words", command=lambda: run_script(os.path.join(base_path, "HaGrid", "Implementation.py"))).pack(pady=10)
    tk.Button(root, text="Alphabets", command=lambda: run_script(os.path.join(base_path, "button.py"))).pack(pady=10)
    tk.Button(root, text="Back", command=init_main_window).pack(pady=10)

def draw():
    base_path = os.path.dirname(os.path.abspath(__file__))
    # The actual folder in the repo is AirCanvas/src/main1.py
    run_script(os.path.join(base_path, "AirCanvas", "src", "main1.py"))

# UI Initialization
def init_main_window():
    clear_frame()
    tk.Button(root, text="Detect", command=detect_options, width=20).pack(pady=10)
    tk.Button(root, text="Draw", command=draw, width=20).pack(pady=10)

def clear_frame():
    for widget in root.winfo_children():
        widget.destroy()

# Create main window
root = tk.Tk()
root.title("Sign Language Interface")
root.geometry("300x200")

init_main_window()
root.mainloop()
