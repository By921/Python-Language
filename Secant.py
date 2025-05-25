import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sympy import symbols, sympify, lambdify

x = symbols('x')

class SecantMethodApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Secant Method Solver")
        self.root.geometry("1000x600")

        self.create_input_section()

        # Create a PanedWindow for resizable layout
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.table_frame = tk.Frame(self.paned_window)
        self.graph_frame = tk.Frame(self.paned_window)

        self.paned_window.add(self.table_frame, minsize=300)
        self.paned_window.add(self.graph_frame, minsize=400)

    def create_input_section(self):
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10, fill='x')

        tk.Label(input_frame, text="Enter Equation f(x):").grid(row=0, column=0, sticky='w')
        self.eq_entry = tk.Entry(input_frame, width=50)
        self.eq_entry.grid(row=0, column=1)
        self.eq_entry.insert(0, "x**3 - 6*x**2 + 11*x - 6")  # Default example

        tk.Label(input_frame, text="Start of Range:").grid(row=1, column=0, sticky='w')
        self.range_start = tk.Entry(input_frame)
        self.range_start.grid(row=1, column=1, sticky='w')
        self.range_start.insert(0, "0")

        tk.Label(input_frame, text="End of Range:").grid(row=2, column=0, sticky='w')
        self.range_end = tk.Entry(input_frame)
        self.range_end.grid(row=2, column=1, sticky='w')
        self.range_end.insert(0, "4")

        tk.Label(input_frame, text="Tolerance (e.g., 1e-6):").grid(row=3, column=0, sticky='w')
        self.tol_entry = tk.Entry(input_frame)
        self.tol_entry.grid(row=3, column=1, sticky='w')
        self.tol_entry.insert(0, "1e-6")

        tk.Label(input_frame, text="Max Iterations:").grid(row=4, column=0, sticky='w')
        self.max_iter_entry = tk.Entry(input_frame)
        self.max_iter_entry.grid(row=4, column=1, sticky='w')
        self.max_iter_entry.insert(0, "100")

        tk.Button(input_frame, text="Find Roots", command=self.run_secant).grid(row=5, columnspan=2, pady=10)

        tk.Label(input_frame, text=(
            "Instructions:\n"
            "- Use 'x' as the variable.\n"
            "- Example: x**3 - 6*x**2 + 11*x - 6\n"
            "- Enter a range covering possible roots."
        ), fg="blue", justify="left").grid(row=6, columnspan=2, pady=10, sticky='w')

    def run_secant(self):
        eq_str = self.eq_entry.get()
        try:
            f_expr = sympify(eq_str)
            f = lambdify(x, f_expr, "numpy")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid equation: {e}")
            return

        try:
            start = float(self.range_start.get())
            end = float(self.range_end.get())
            tol = float(self.tol_entry.get())
            max_iter = int(self.max_iter_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric input.")
            return

        roots = []
        tables = []

        scan_points = np.linspace(start, end, 100)
        for i in range(len(scan_points) - 1):
            x0, x1 = scan_points[i], scan_points[i + 1]
            try:
                table, root = self.secant(f, x0, x1, tol, max_iter)
                if not any(np.isclose(root, r, atol=tol) for r in roots):
                    roots.append(root)
                    tables.append(table)
            except Exception:
                continue

        if not roots:
            messagebox.showinfo("No Roots", "No roots found in the given range.")
            return

        self.show_results(f, f_expr, roots, tables, start, end)

    def secant(self, f, x0, x1, tol, max_iter):
        table = []
        for i in range(max_iter):
            if f(x1) - f(x0) == 0:
                raise Exception("Divide by zero in Secant formula.")
            x2 = x1 - f(x1) * ((x1 - x0) / (f(x1) - f(x0)))
            table.append([i + 1, x0, x1, x2, f(x2)])
            if abs(x2 - x1) < tol:
                return table, x2
            x0, x1 = x1, x2
        raise Exception("Secant method did not converge.")

    def show_results(self, f, f_expr, roots, tables, start, end):
        # Clear previous contents
        for frame in [self.table_frame, self.graph_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

        # --- Table ---
        last_table = tables[-1]
        columns = ("Iter", "x0", "x1", "x2", "f(x2)")
        tree = ttk.Treeview(self.table_frame, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
        for row in last_table:
            tree.insert("", "end", values=[f"{v:.6g}" for v in row])
        tree.pack(fill="both", expand=True)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 4))
        x_vals = np.linspace(start, end, 400)
        y_vals = f(x_vals)
        ax.plot(x_vals, y_vals, label=f"f(x) = {f_expr}", color='blue')

        for i, root in enumerate(roots):
            ax.plot(root, f(root), 'ro')
            ax.annotate(f"Root {i+1}: {root:.4f}", (root, f(root)), textcoords="offset points", xytext=(0, 10), ha='center')

        ax.axhline(0, color='gray', linestyle='--')
        ax.legend()
        ax.set_title("Secant Method Root Finding")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = SecantMethodApp(root)
    root.mainloop()
