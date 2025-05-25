import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import symbols, sympify, lambdify


class BisectionMethodApp(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Bisection Method")
        self.geometry("1000x600")

        self.create_widgets()

    def bisection(self, f, a, b, tol=1e-5, max_iter=100):
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            return None, []  # No root
        table = []
        for i in range(max_iter):
            c = (a + b) / 2
            fc = f(c)
            table.append([i + 1, a, b, c, fc])
            if abs(fc) < tol or abs(b - a) < tol:
                return c, table
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        return None, table

    def find_roots(self, f, a, b, step, tol=1e-5):
        roots = []
        tables = []
        x_vals = np.arange(a, b, step)
        for i in range(len(x_vals) - 1):
            x1, x2 = x_vals[i], x_vals[i + 1]
            root, table = self.bisection(f, x1, x2, tol)
            if root is not None and all(abs(root - r) > tol for r in roots):
                roots.append(root)
                tables.append(table)
        return roots, tables

    def plot_function_and_roots(self, f, a, b, roots):
        for widget in self.frame_plot.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 4))
        x_vals = np.linspace(a, b, 1000)
        y_vals = f(x_vals)

        ax.plot(x_vals, y_vals, label='f(x)')
        ax.axhline(0, color='black', linewidth=0.5)

        for i, root in enumerate(roots):
            ax.plot(root, f(root), 'ro')
            ax.text(root, f(root), f"Root {i+1}\n{root:.5f}", fontsize=8,
                    ha='center', va='bottom')
        ax.legend()
        ax.set_title("Function and Roots")
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def display_table(self, table):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for i, row in enumerate(table):
            self.tree.insert('', tk.END, values=[f"{v:.6f}" if isinstance(v, float) else v for v in row])

    def calculate(self):
        try:
            expr_str = self.entry_eq.get()
            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            step = float(self.entry_step.get())

            x = symbols('x')
            expr = sympify(expr_str)
            f = lambdify(x, expr, 'numpy')

            roots, tables = self.find_roots(f, a, b, step)

            if not roots:
                messagebox.showinfo("Result", "No roots found in the interval.")
                return

            self.plot_function_and_roots(f, a, b, roots)

            self.label_result.config(text="Found Roots:\n" + "\n".join([f"Root {i+1}: {r:.6f}" for i, r in enumerate(roots)]))

            # Show the table for the first root
            self.display_table(tables[0])

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_widgets(self):
        frame_input = tk.Frame(self)
        frame_input.pack(pady=10)

        tk.Label(frame_input, text="f(x):").grid(row=0, column=0)
        self.entry_eq = tk.Entry(frame_input, width=30)
        self.entry_eq.insert(0, "x**3 - 6*x**2 + 11*x - 6")
        self.entry_eq.grid(row=0, column=1)

        tk.Label(frame_input, text="a:").grid(row=0, column=2)
        self.entry_a = tk.Entry(frame_input, width=5)
        self.entry_a.insert(0, "0")
        self.entry_a.grid(row=0, column=3)

        tk.Label(frame_input, text="b:").grid(row=0, column=4)
        self.entry_b = tk.Entry(frame_input, width=5)
        self.entry_b.insert(0, "5")
        self.entry_b.grid(row=0, column=5)

        tk.Label(frame_input, text="Step:").grid(row=0, column=6)
        self.entry_step = tk.Entry(frame_input, width=5)
        self.entry_step.insert(0, "0.5")
        self.entry_step.grid(row=0, column=7)

        tk.Button(frame_input, text="Calculate", command=self.calculate).grid(row=0, column=8, padx=10)

        frame_main = tk.Frame(self)
        frame_main.pack(fill=tk.BOTH, expand=True)

        self.frame_plot = tk.LabelFrame(frame_main, text="Graph")
        self.frame_plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_table = tk.LabelFrame(frame_main, text="Bisection Table (First Root)")
        frame_table.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        cols = ["Iter", "a", "b", "c", "f(c)"]
        self.tree = ttk.Treeview(frame_table, columns=cols, show='headings', height=25)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80)
        self.tree.pack(fill=tk.Y)

        self.label_result = tk.Label(self, text="", font=("Arial", 12), justify=tk.LEFT)
        self.label_result.pack(pady=5)
