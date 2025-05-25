import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import sympify, Symbol, lambdify
import pandas as pd

class FalsePositionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("False Position Method Solver")
        self.root.geometry("900x700")

        instr_text = (
            "Instructions:\n"
            "- Enter a valid function of x (e.g. x**3 - x - 2).\n"
            "- Enter the interval [a, b] to search for roots.\n"
            "- Enter the tolerance for convergence.\n"
            "- The program will scan the interval to find multiple roots.\n"
            "- Click 'Find Roots' to run the method.\n"
            "- The results will show in table and graph form."
        )
        tk.Label(root, text=instr_text, justify=tk.LEFT).pack(pady=5)

        input_frame = tk.Frame(root)
        input_frame.pack(pady=5)

        tk.Label(input_frame, text="Function f(x):").grid(row=0, column=0, sticky=tk.W)
        self.func_entry = tk.Entry(input_frame, width=40)
        self.func_entry.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="Interval start (a):").grid(row=1, column=0, sticky=tk.W)
        self.a_entry = tk.Entry(input_frame, width=15)
        self.a_entry.grid(row=1, column=1, sticky=tk.W, padx=5)

        tk.Label(input_frame, text="Interval end (b):").grid(row=2, column=0, sticky=tk.W)
        self.b_entry = tk.Entry(input_frame, width=15)
        self.b_entry.grid(row=2, column=1, sticky=tk.W, padx=5)

        tk.Label(input_frame, text="Tolerance (e.g. 1e-5):").grid(row=3, column=0, sticky=tk.W)
        self.tol_entry = tk.Entry(input_frame, width=15)
        self.tol_entry.grid(row=3, column=1, sticky=tk.W, padx=5)

        tk.Label(input_frame, text="Subinterval divisions:").grid(row=4, column=0, sticky=tk.W)
        self.subdiv_entry = tk.Entry(input_frame, width=15)
        self.subdiv_entry.insert(0, "50")
        self.subdiv_entry.grid(row=4, column=1, sticky=tk.W, padx=5)

        tk.Button(root, text="Find Roots", command=self.find_roots).pack(pady=10)

        # Table Frame
        self.table_frame = tk.Frame(root)
        self.table_frame.pack(fill=tk.X, pady=(5, 0))

        # Graph Frame
        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def false_position(self, f, a, b, tol, max_iter=100):
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            return None, []
        iterations = []
        for i in range(1, max_iter + 1):
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            iterations.append({
                "Iteration": i, "a": a, "b": b, "c": c,
                "f(c)": fc, "f(a)": fa, "f(b)": fb,
                "Error": abs(fc)
            })
            if abs(fc) < tol:
                return c, iterations
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return None, iterations

    def find_roots(self):
        func_str = self.func_entry.get()
        try:
            x = Symbol('x')
            expr = sympify(func_str)
            f = lambdify(x, expr, 'numpy')
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid function:\n{e}")
            return

        try:
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            if b <= a:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Invalid interval [a, b].")
            return

        try:
            tol = float(self.tol_entry.get())
            if tol <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Invalid tolerance.")
            return

        try:
            subdivisions = int(self.subdiv_entry.get())
            if subdivisions < 2:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Invalid subdivisions.")
            return

        # Clear table and graph
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self.ax.clear()

        xs = np.linspace(a, b, subdivisions + 1)
        roots, tables = [], []

        for i in range(subdivisions):
            a_i, b_i = xs[i], xs[i + 1]
            try:
                fa_i, fb_i = f(a_i), f(b_i)
            except Exception as e:
                messagebox.showerror("Evaluation Error", str(e))
                return

            if np.sign(fa_i) * np.sign(fb_i) <= 0:
                root, iterations = self.false_position(f, a_i, b_i, tol)
                if root is not None and not any(abs(root - r) < tol * 10 for r in roots):
                    roots.append(root)
                    tables.append(iterations)

        if not roots:
            messagebox.showinfo("Result", "No roots found.")
            return

        # Tables
        for idx, iterations in enumerate(tables):
            df = pd.DataFrame(iterations)
            lbl = tk.Label(self.table_frame, text=f"Root {idx + 1} iterations:")
            lbl.pack(anchor=tk.W, pady=(10, 0))
            tree = ttk.Treeview(self.table_frame, columns=list(df.columns), show='headings', height=8)
            for col in df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=80, anchor=tk.CENTER)
            for _, row in df.iterrows():
                values = [f"{v:.6g}" if isinstance(v, float) else v for v in row]
                tree.insert("", "end", values=values)
            tree.pack(fill=tk.X, pady=(0, 10))

        # Plotting
        try:
            plot_x = np.linspace(a, b, 1000)
            plot_y = f(plot_x)
        except Exception as e:
            messagebox.showerror("Plotting Error", f"Error plotting: {e}")
            return

        self.ax.plot(plot_x, plot_y, label="f(x)")
        self.ax.axhline(0, color="black", linewidth=0.8)
        for i, root in enumerate(roots):
            self.ax.plot(root, 0, 'ro')
            self.ax.annotate(f"Root {i+1}\n{root:.6g}", (root, 0),
                             textcoords="offset points", xytext=(0, 10),
                             ha='center', fontsize=8, color='red')

        self.ax.set_title("Function Plot with Roots")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw_idle()  # Redraw properly

if __name__ == "__main__":
    root = tk.Tk()
    app = FalsePositionApp(root)
    root.mainloop()
