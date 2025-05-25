import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sympy as sp
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas as pdf_canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NewtonRaphsonApp:
    def __init__(self, parent):
        self.tables = []
        self.root = parent
        self.root.title("Newton-Raphson Method - GUI")
        self.root.geometry("1100x700")

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style(self.root)
        style.theme_use("default")

        # Main vertical paned window
        vertical_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        vertical_paned.pack(fill="both", expand=True, padx=10, pady=10)

        # Horizontal paned window for graph & table
        paned = ttk.PanedWindow(vertical_paned, orient=tk.HORIZONTAL)

        # Graph frame
        self.graph_frame = ttk.Frame(paned, width=600, height=500)
        paned.add(self.graph_frame, weight=3)

        # Table frame
        self.table_frame = ttk.Frame(paned, width=400, height=500)
        paned.add(self.table_frame, weight=2)

        vertical_paned.add(paned, weight=7)

        # Bottom paned window for controls and instructions
        bottom_paned = ttk.PanedWindow(vertical_paned, orient=tk.HORIZONTAL)

        # Control frame
        control_frame = ttk.Frame(bottom_paned, width=600, height=150)
        bottom_paned.add(control_frame, weight=3)

        # Instructions
        instruction_container = ttk.Frame(bottom_paned, width=400, height=150)
        bottom_paned.add(instruction_container, weight=2)

        vertical_paned.add(bottom_paned, weight=2)

        # Control widgets
        tk.Label(control_frame, text="Function f(x):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.entry_eq = tk.Entry(control_frame, width=40)
        self.entry_eq.grid(row=0, column=1, columnspan=4, sticky='w', padx=5, pady=5)
        self.entry_eq.insert(0, "x**3 - 6*x**2 + 11*x - 6")

        tk.Label(control_frame, text="Range Start:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.entry_start = tk.Entry(control_frame, width=10)
        self.entry_start.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        self.entry_start.insert(0, "-5")

        tk.Label(control_frame, text="End:").grid(row=1, column=2, sticky='w', padx=5, pady=5)
        self.entry_end = tk.Entry(control_frame, width=10)
        self.entry_end.grid(row=1, column=3, sticky='w', padx=5, pady=5)
        self.entry_end.insert(0, "10")

        ttk.Button(control_frame, text="Find Roots", command=self.run_newton_gui).grid(row=1, column=4, padx=5, pady=5)
        ttk.Button(control_frame, text="Export Excel", command=self.export_to_excel).grid(row=1, column=5, padx=5, pady=5)
        ttk.Button(control_frame, text="Export PDF", command=self.export_to_pdf).grid(row=1, column=6, padx=5, pady=5)

        # Instruction text
        scrollbar = ttk.Scrollbar(instruction_container, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        instruction_text_widget = tk.Text(instruction_container, wrap="word", height=8, yscrollcommand=scrollbar.set)
        instruction_text_widget.pack(fill="both", expand=True)
        scrollbar.config(command=instruction_text_widget.yview)

        instruction_text = (
            "Instructions:\n"
            "- Enter function using 'x'. Use ** for powers.\n"
            "- Example: x**3 - 6*x**2 + 11*x - 6\n"
            "- Range defines the interval to search for roots.\n"
            "- Click 'Find Roots' to execute Newton-Raphson.\n"
            "- Use export buttons to save data.\n"
        )
        instruction_text_widget.insert("1.0", instruction_text)
        instruction_text_widget.config(state="disabled")

    def newton_raphson(self, f_expr, x0, tol=1e-6, max_iter=50):
        x = sp.symbols('x')
        f = sp.sympify(f_expr)
        f_prime = sp.diff(f, x)
        f_lambda = sp.lambdify(x, f)
        f_prime_lambda = sp.lambdify(x, f_prime)
        iterations = []
        for i in range(max_iter):
            f_x0 = f_lambda(x0)
            f_prime_x0 = f_prime_lambda(x0)
            if f_prime_x0 == 0:
                break
            x1 = x0 - f_x0 / f_prime_x0
            error = abs(x1 - x0)
            iterations.append((i+1, x0, f_x0, f_prime_x0, x1, error))
            if error < tol:
                break
            x0 = x1
        return iterations

    def find_multiple_roots(self, f_expr, start, end, steps=200):
        x_vals = np.linspace(start, end, steps)
        x = sp.symbols('x')
        f = sp.sympify(f_expr)
        f_lambdified = sp.lambdify(x, f)
        y_vals = f_lambdified(x_vals)
        roots = []
        all_iterations = []
        for i in range(len(x_vals) - 1):
            if y_vals[i] * y_vals[i + 1] < 0:
                try:
                    mid = (x_vals[i] + x_vals[i + 1]) / 2
                    root_iters = self.newton_raphson(f_expr, mid)
                    if root_iters:
                        root = round(root_iters[-1][4], 6)
                        if not any(abs(root - r) < 1e-4 for r in roots):
                            roots.append(root)
                            all_iterations.append((root, root_iters))
                except:
                    pass
        return roots, all_iterations

    def plot_graph(self, f_expr, roots, start, end):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        x = sp.symbols('x')
        f = sp.sympify(f_expr)
        f_lambda = sp.lambdify(x, f)
        x_vals = np.linspace(start, end, 1000)
        y_vals = f_lambda(x_vals)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(x_vals, y_vals, label=f"f(x) = {f_expr}", color='blue')
        ax.axhline(0, color='black', linestyle='--')
        for i, r in enumerate(roots):
            ax.plot(r, 0, 'ro')
            ax.annotate(f"Root {i+1}: {r:.4f}", (r, 0), textcoords="offset points", xytext=(0,10), ha='center')
        ax.set_title("Root Plot")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def display_tables(self, root_data):
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self.tables.clear()
        for root, data in root_data:
            label = ttk.Label(self.table_frame, text=f"Root at x = {root:.6f}", font=("Arial", 12, "bold"))
            label.pack(pady=(10, 0))
            cols = ['Iteration', 'x_n', 'f(x_n)', "f'(x_n)", 'x_(n+1)', 'Error']
            df = pd.DataFrame(data, columns=cols)
            self.tables.append((f"Root_{root:.4f}", df))
            tree = ttk.Treeview(self.table_frame, columns=cols, show="headings", height=min(15, len(df)))
            for col in cols:
                tree.heading(col, text=col)
                tree.column(col, anchor=tk.CENTER, width=110)
            for row in df.itertuples(index=False):
                tree.insert("", "end", values=row)
            tree.pack(fill="both", padx=5, pady=5, expand=True)
            ttk.Separator(self.table_frame, orient='horizontal').pack(fill='x', pady=5)

    def export_to_excel(self):
        if not self.tables:
            messagebox.showinfo("No data", "Run the method first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx")
        if not file_path:
            return
        with pd.ExcelWriter(file_path) as writer:
            for name, df in self.tables:
                df.to_excel(writer, sheet_name=name, index=False)
        messagebox.showinfo("Saved", f"Data saved to {file_path}")

    def export_to_pdf(self):
        if not self.tables:
            messagebox.showinfo("No data", "Run the method first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf")
        if not file_path:
            return
        c = pdf_canvas.Canvas(file_path)
        text = c.beginText(40, 800)
        for name, df in self.tables:
            text.textLine(name)
            text.textLine("")
            for _, row in df.iterrows():
                text.textLine("  ".join(str(v)[:12] for v in row))
            text.textLine("")
        c.drawText(text)
        c.showPage()
        c.save()
        messagebox.showinfo("Saved", f"PDF saved to {file_path}")

    def run_newton_gui(self):
        f_expr = self.entry_eq.get().strip()
        try:
            start = float(self.entry_start.get())
            end = float(self.entry_end.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Range must be numbers.")
            return
        if not f_expr:
            messagebox.showerror("Missing Input", "Enter a function.")
            return
        try:
            roots, all_iterations = self.find_multiple_roots(f_expr, start, end)
            if roots:
                self.display_tables(all_iterations)
                self.plot_graph(f_expr, roots, start, end)
            else:
                messagebox.showinfo("No Roots", "No roots found in range.")
                for widget in self.table_frame.winfo_children():
                    widget.destroy()
                for widget in self.graph_frame.winfo_children():
                    widget.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))
