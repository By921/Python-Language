import tkinter as tk
from tkinter import ttk
from incremental_method import IncrementalSearchApp
from Bisection import BisectionMethodApp
from FalsePosition import FalsePositionApp
from newton_gui import NewtonRaphsonApp
from Secant import SecantMethodApp

class windowNumericalUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Numerical Methods")
        self.root.geometry("400x480")
        self.root.minsize(380, 460)
        self.root.configure(bg="#f7f9fc")

        self.build_ui()
        self.root.mainloop()

    def build_ui(self):
        container = tk.Frame(self.root, bg="#f7f9fc")
        container.pack(expand=True, pady=40, padx=40)

        title = tk.Label(container, text="Numerical Methods",
                         font=("Segoe UI", 22, "bold"),
                         fg="#2c3e50", bg="#f7f9fc")
        title.pack(pady=(0, 25))

        methods = [
            ("Incremental Method", self.open_incremental_window),
            ("Bisection Method", self.open_bisection_window),
            ("False Position Method", self.open_falseposition_window),
            ("Newton-Raphson Method", self.open_newtonraphson_window),
            ("Secant Method", self.open_secant_window)
        ]

        style = ttk.Style()
        style.configure("Method.TButton",
                        font=("Segoe UI", 12),
                        padding=(15, 10),
                        foreground="#2c3e50",
                        background="#ffffff")
        style.map("Method.TButton",
                  background=[("active", "#d0d7e3"), ("!active", "#f5f7fa")])

        for text, cmd in methods:
            btn = ttk.Button(container, text=text, command=cmd, style="Method.TButton")
            btn.pack(pady=8, fill="x")

    def open_incremental_window(self):
        IncrementalSearchApp(self.root)

    def open_bisection_window(self):
        BisectionMethodApp(self.root)

    def open_falseposition_window(self):
        new_window = tk.Toplevel(self.root)
        FalsePositionApp(new_window)

    def open_newtonraphson_window(self):
        new_window = tk.Toplevel(self.root)
        NewtonRaphsonApp(new_window)

    def open_secant_window(self):
        new_window = tk.Toplevel(self.root)
        SecantMethodApp(new_window)

if __name__ == "__main__":
    windowNumericalUI()
