import tkinter as tk
from tkinter import ttk, messagebox, font
from sympy import symbols, sympify, lambdify, pi, E, exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
import re


def preprocess_function(expr_str):
    expr_str = expr_str.replace('pi', 'π').replace('Pi', 'π').replace('PI', 'π')

    # Handle e^ expressions (e.g., e^-x, e^(x+1))
    expr_str = re.sub(r'e\^\(?(.+?)\)?', r'exp(\1)', expr_str)

    # Handle trigonometric exponents (e.g., sin^2(x) -> (sin(x))**2)
    expr_str = re.sub(r'(sin|cos|tan|cot|sec|csc)\^(\d+)\((.+?)\)', r'(\1(\3))**\2', expr_str)

    # Replace ^ with **
    expr_str = expr_str.replace('^', '**')

    # Handle log base 10 (convert log to log10)
    expr_str = re.sub(r'\blog\(', 'log(', expr_str)  # SymPy uses ln for log
    return expr_str


def insert_multiplication_sign(expr):
    # Between number and variable/function: 2x -> 2*x, 2sin(x) -> 2*sin(x)
    expr = re.sub(r'(\d+)([a-zA-Z\(])', r'\1*\2', expr)

    # Between closing parenthesis and opening: (x+1)(x-1) -> (x+1)*(x-1)
    expr = re.sub(r'(\))(\()', r'\1*\2', expr)

    # Between variable and opening parenthesis: x(x+1) -> x*(x+1)
    expr = re.sub(r'([a-zA-Zπ])(\()', r'\1*\2', expr)

    # Between closing parenthesis and variable: (x+1)y -> (x+1)*y
    expr = re.sub(r'(\))([a-zA-Zπ])', r'\1*\2', expr)

    # Between variables/constants: xy -> x*y, πx -> π*x
    expr = re.sub(r'([a-zA-Zπ])(?=[a-zA-Zπ])', r'\1*', expr)
    return expr


def safe_eval(f_expr, val):
    """
    Safely evaluate a symbolic expression at a given value.
    """
    x = symbols('x')
    subs = {x: val}

    # Handle special constants
    constants = {
        'π': pi,
        'pi': pi,
        'e': E
    }

    # Add all constants to substitution dictionary
    for symbol in f_expr.free_symbols:
        symbol_name = str(symbol)
        if symbol_name in constants:
            subs[symbol] = constants[symbol_name]
        elif symbol != x:
            subs[symbol] = 0

    return f_expr.evalf(subs=subs)


def find_appropriate_bounds(expr, x_symbol):
    """
    Find appropriate bounds for plotting a function.
    Returns (x_min, x_max, y_min, y_max)
    """
    # Create a dictionary for constants
    constants = {
        'π': pi,
        'pi': pi,
        'e': E
    }

    # Substitute constants in the expression
    subs_dict = {symb: constants.get(str(symb), 0) for symb in expr.free_symbols if symb != x_symbol}
    expr_with_constants = expr.subs(subs_dict)

    # Create a lambda function for evaluation
    f = lambdify(x_symbol, expr_with_constants, modules=["numpy", "sympy"])

    # Try different ranges to find where the function is well-behaved
    test_ranges = [
        (-5, 5),  # Standard range
        (-10, 10),  # Wider range
        (-2 * pi, 2 * pi),  # Good for trigonometric functions
        (-1, 5),  # Positive-focused range
        (-20, 20)  # Very wide range
    ]

    best_range = None
    best_score = float('inf')  # Lower is better

    for x_min, x_max in test_ranges:
        try:
            # Sample points in this range
            xs = np.linspace(x_min, x_max, 100)
            ys = f(xs)

            # Filter out infinities and NaNs
            valid_ys = ys[~np.isinf(ys) & ~np.isnan(ys)]

            if len(valid_ys) < 10:  # Not enough valid points
                continue

            # Calculate range statistics
            y_range = np.max(valid_ys) - np.min(valid_ys)

            # Skip if range is too large (likely has asymptotes)
            if y_range > 1000:
                continue

            # Calculate a score (lower is better)
            # We prefer ranges with more valid points and reasonable y-range
            score = y_range / len(valid_ys)

            if score < best_score:
                best_score = score
                y_min = np.min(valid_ys) - 0.1 * y_range
                y_max = np.max(valid_ys) + 0.1 * y_range
                best_range = (x_min, x_max, y_min, y_max)

        except Exception:
            continue

    # If no good range found, return a default
    if best_range is None:
        return (-10, 10, -10, 10)

    return best_range


# Custom toolbar with only the features we want
class CustomNavigationToolbar(NavigationToolbar2Tk):
    # Only keep the save, pan, and home buttons
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                 t[0] in ('Home', 'Pan', 'Save')]

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)


class IncrementalSearchApp(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Incremental Search Method")
        self.geometry("1000x700")
        self.configure(bg="#f5f5f5")

        # Set custom fonts
        self.title_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=10)
        self.button_font = font.Font(family="Helvetica", size=10, weight="bold")

        # Plot settings
        self.plot_color = "#1f77b4"  # Default plot color
        self.grid_on = True
        self.show_points = True
        self.show_root_area = True
        self.view_mode = "table"  # Initial view is table

        # Main container
        self.main_frame = ttk.Frame(self, padding="20 20 20 20")
        self.main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="Incremental Search Method",
            font=self.title_font
        )
        title_label.pack(pady=(0, 15))

        # Input panel
        input_panel = ttk.LabelFrame(self.main_frame, text="Input Parameters", padding="10 10 10 10")
        input_panel.pack(fill="x", pady=(0, 10))

        # Create a horizontal layout for inputs
        input_frame = ttk.Frame(input_panel)
        input_frame.pack(fill="x")

        # Left side - Function input
        func_frame = ttk.Frame(input_frame)
        func_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ttk.Label(
            func_frame,
            text="Function f(x):",
            font=self.label_font
        ).pack(anchor="w")

        self.func_entry = ttk.Entry(func_frame, width=30)
        self.func_entry.pack(fill="x", pady=(2, 0))
        self.func_entry.insert(0, "x^2 - 4")

        # Function examples and help
        examples_frame = ttk.Frame(func_frame)
        examples_frame.pack(fill="x", pady=(2, 0))

        ttk.Label(
            examples_frame,
            text="Examples:",
            font=("Helvetica", 8, "italic")
        ).pack(side="left", anchor="w")

        # Example buttons
        example_functions = [
            ("x^2 - 4", "x^2 - 4"),
            ("sin(x)", "sin(x)"),
            ("e^-x - x", "e^-x - x"),
            ("ln(x)", "ln(x)"),
            ("pi*sin(x)", "pi*sin(x)")
        ]

        for label, func in example_functions:
            example_btn = ttk.Button(
                examples_frame,
                text=label,
                style="Example.TButton",
                width=8,
                command=lambda f=func: self.set_example_function(f)
            )
            example_btn.pack(side="left", padx=2)

        # Create a style for example buttons
        style = ttk.Style()
        style.configure("Example.TButton", font=("Helvetica", 7))

        # Right side - Parameters
        params_frame = ttk.Frame(input_frame)
        params_frame.pack(side="left", fill="x", expand=True)

        # Grid for parameters
        param_labels = ["Start x₀:", "Increment Δx:", "Max Iterations:"]
        param_entries = ["start_entry", "increment_entry", "max_iter_entry"]
        param_defaults = ["1", "0.5", "20"]

        for i, (label, entry_name, default) in enumerate(zip(param_labels, param_entries, param_defaults)):
            ttk.Label(params_frame, text=label, font=self.label_font).grid(
                row=i, column=0, sticky="w", pady=5
            )
            entry = ttk.Entry(params_frame, width=15)
            entry.grid(row=i, column=1, sticky="w", padx=(10, 0), pady=5)
            entry.insert(0, default)
            setattr(self, entry_name, entry)

        # Auto bounds checkbox
        self.auto_bounds_var = tk.BooleanVar(value=True)
        auto_bounds_check = ttk.Checkbutton(
            params_frame,
            text="Auto Find Bounds",
            variable=self.auto_bounds_var
        )
        auto_bounds_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=5)

        # Button frame
        button_frame = ttk.Frame(input_panel)
        button_frame.pack(fill="x", pady=(10, 0))

        # Compute button
        compute_button = ttk.Button(
            button_frame,
            text="Compute",
            command=self.compute,
            style="Accent.TButton"
        )
        compute_button.pack(side="left", padx=(0, 10))

        # View toggle button
        self.view_button = ttk.Button(
            button_frame,
            text="View Graph",
            command=self.toggle_view,
            style="View.TButton"
        )
        self.view_button.pack(side="left")

        # Plot Only button
        plot_button = ttk.Button(
            button_frame,
            text="Plot Function",
            command=self.plot_only,
            style="Plot.TButton"
        )
        plot_button.pack(side="left", padx=(10, 0))

        # Help button
        help_button = ttk.Button(
            button_frame,
            text="Help",
            command=self.show_help,
            style="Help.TButton"
        )
        help_button.pack(side="right")

        # Create styles for the buttons
        style.configure("Accent.TButton", font=self.button_font)
        style.configure("View.TButton", font=self.button_font)
        style.configure("Plot.TButton", font=self.button_font)
        style.configure("Help.TButton", font=self.button_font)

        # Create a container frame for the views
        self.view_container = ttk.Frame(self.main_frame)
        self.view_container.pack(fill="both", expand=True)

        # Create the table view
        self.create_table_view()

        # Create the graph view (but don't pack it yet)
        self.create_graph_view()

        # Show the table view initially
        self.table_frame.pack(fill="both", expand=True)

        # Store results for plot updates
        self.results = []
        self.expr = None
        self.x_symbol = symbols("x")

        # Store zoom state
        self.zoom_scale = 1.1  # Zoom factor

        # Status bar for feedback
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor="w"
        )
        self.status_bar.pack(side="bottom", fill="x")

    def set_example_function(self, func):
        """Set an example function in the entry field."""
        self.func_entry.delete(0, tk.END)
        self.func_entry.insert(0, func)
        self.status_var.set(f"Function set to: {func}")

    def show_help(self):
        """Show help information about function syntax."""
        help_text = """
Function Syntax Help:

Basic Operations:
  + Addition: x + 2
  - Subtraction: x - 2
  * Multiplication: x * 2 (or simply 2x)
  / Division: x / 2
  ^ or ** Exponentiation: x^2 or x**2

Functions:
  sin(x), cos(x), tan(x) - Trigonometric functions
  asin(x), acos(x), atan(x) - Inverse trigonometric functions
  exp(x) or e^x - Exponential function
  ln(x) or log(x) - Natural logarithm
  sqrt(x) - Square root

Constants:
  pi or π - The constant π (3.14159...)
  e - The constant e (2.71828...)

Examples:
  x^2 - 4
  sin(x) + cos(x)
  e^-x - x
  ln(x^2 + 1)
  pi*sin(x)

Tips:
  - For negative exponents, use parentheses: e^(-x)
  - You can omit multiplication signs: 2x is the same as 2*x
  - Use parentheses to group expressions: sin(x^2 + 1)
        """

        help_window = tk.Toplevel(self)
        help_window.title("Function Syntax Help")
        help_window.geometry("500x500")

        # Add a text widget with scrollbar
        text_frame = ttk.Frame(help_window, padding="10 10 10 10")
        text_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text_widget = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set)
        text_widget.pack(fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert("1.0", help_text)
        text_widget.config(state="disabled")  # Make it read-only

    def create_table_view(self):
        # Table view
        self.table_frame = ttk.LabelFrame(self.view_container, text="Results", padding="10 10 10 10")

        # Scrollbars
        tree_scroll_y = ttk.Scrollbar(self.table_frame)
        tree_scroll_y.pack(side="right", fill="y")

        tree_scroll_x = ttk.Scrollbar(self.table_frame, orient="horizontal")
        tree_scroll_x.pack(side="bottom", fill="x")

        # Results Table
        columns = ("Iteration", "x_l", "Δx", "x_u", "f(x_l)", "f(x_u)", "Product", "Remarks")
        self.tree = ttk.Treeview(
            self.table_frame,
            columns=columns,
            show="headings",
            height=15,
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )

        # Configure scrollbars
        tree_scroll_y.config(command=self.tree.yview)
        tree_scroll_x.config(command=self.tree.xview)

        # Configure columns
        for col in columns:
            self.tree.heading(col, text=col)
            width = 70 if col in ["Iteration", "Δx", "Remarks"] else 90
            self.tree.column(col, width=width, anchor="center")

        self.tree.pack(fill="both", expand=True)

    def create_graph_view(self):
        # Graph view
        self.graph_frame = ttk.LabelFrame(self.view_container, text="Function Plot", padding="10 10 10 10")

        # Plot controls frame
        plot_controls = ttk.Frame(self.graph_frame)
        plot_controls.pack(fill="x", pady=(0, 5))

        # Grid toggle
        self.grid_var = tk.BooleanVar(value=True)
        grid_check = ttk.Checkbutton(
            plot_controls,
            text="Show Grid",
            variable=self.grid_var,
            command=self.update_plot_settings
        )
        grid_check.pack(side="left", padx=5)

        # Points toggle
        self.points_var = tk.BooleanVar(value=True)
        points_check = ttk.Checkbutton(
            plot_controls,
            text="Show Points",
            variable=self.points_var,
            command=self.update_plot_settings
        )
        points_check.pack(side="left", padx=5)

        # Root area toggle
        self.root_area_var = tk.BooleanVar(value=True)
        root_area_check = ttk.Checkbutton(
            plot_controls,
            text="Highlight Root Area",
            variable=self.root_area_var,
            command=self.update_plot_settings
        )
        root_area_check.pack(side="left", padx=5)

        # Reset view button
        reset_view_button = ttk.Button(
            plot_controls,
            text="Reset View",
            command=self.reset_plot_view
        )
        reset_view_button.pack(side="left", padx=5)

        # Create an enhanced plot
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor='#f8f9fa')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("x", fontsize=12, fontweight='bold')
        self.ax.set_ylabel("f(x)", fontsize=12, fontweight='bold')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')

        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()

        # Add custom toolbar with limited features
        self.toolbar_frame = ttk.Frame(self.graph_frame)
        self.toolbar_frame.pack(side="top", fill="x")
        self.toolbar = CustomNavigationToolbar(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Add mouse position display
        self.coordinate_label = ttk.Label(self.graph_frame, text="Coordinates: x=0.000, y=0.000")
        self.coordinate_label.pack(side="bottom", anchor="e", padx=5, pady=2)

        # Connect mouse events
        self.canvas.mpl_connect('motion_notify_event', self.update_coordinates)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        # Get the current x and y limits
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        # Get event location
        xdata = event.xdata
        ydata = event.ydata

        # Return if cursor is outside the plot
        if xdata is None or ydata is None:
            return

        # Get the directions
        if event.button == 'up':
            # Zoom in
            scale_factor = 1 / self.zoom_scale
        elif event.button == 'down':
            # Zoom out
            scale_factor = self.zoom_scale
        else:
            # No zoom for other buttons
            return

        # Calculate new limits
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        # Calculate new center based on mouse position
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        # Set new limits
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        # Update the plot
        self.canvas.draw_idle()

    def toggle_view(self):
        if self.view_mode == "table":
            # Switch to graph view
            self.table_frame.pack_forget()
            self.graph_frame.pack(fill="both", expand=True)
            self.view_mode = "graph"
            self.view_button.config(text="View Table")
            # Update the plot
            self.update_plot()
        else:
            # Switch to table view
            self.graph_frame.pack_forget()
            self.table_frame.pack(fill="both", expand=True)
            self.view_mode = "table"
            self.view_button.config(text="View Graph")

    def update_coordinates(self, event):
        if event.inaxes:
            self.coordinate_label.config(text=f"Coordinates: x={event.xdata:.3f}, y={event.ydata:.3f}")

    def update_plot_settings(self):
        self.grid_on = self.grid_var.get()
        self.show_points = self.points_var.get()
        self.show_root_area = self.root_area_var.get()
        self.update_plot()

    def reset_plot_view(self):
        if hasattr(self, 'expr') and self.expr is not None:
            self.ax.clear()
            self.update_plot(reset=True)
        else:
            self.ax.clear()
            self.ax.set_xlabel("x", fontsize=12, fontweight='bold')
            self.ax.set_ylabel("f(x)", fontsize=12, fontweight='bold')
            self.ax.grid(self.grid_on, linestyle='--', alpha=0.7)
            self.ax.spines['left'].set_position('zero')
            self.ax.spines['bottom'].set_position('zero')
            self.ax.spines['right'].set_color('none')
            self.ax.spines['top'].set_color('none')
            self.canvas.draw()

    def plot_only(self):
        """Plot the function without performing incremental search."""
        # Get the function string and preprocess it
        func_str = self.func_entry.get().strip()

        # Update status
        self.status_var.set(f"Processing function: {func_str}")

        # Preprocess the function string
        func_str = preprocess_function(func_str)

        # Insert multiplication signs where implied
        func_str = insert_multiplication_sign(func_str)

        try:
            x = symbols("x")
            self.x_symbol = x

            # Add special constants
            constants = {
                'π': pi,
                'pi': pi,
                'e': E
            }

            # Parse the expression with sympy
            self.expr = sympify(func_str, locals=constants)

            # Update status
            self.status_var.set(f"Function parsed successfully: {self.expr}")

            # Switch to graph view if not already there
            if self.view_mode != "graph":
                self.toggle_view()

            # Find appropriate bounds if auto-bounds is enabled
            if self.auto_bounds_var.get():
                bounds = find_appropriate_bounds(self.expr, self.x_symbol)
                self.update_plot(reset=True, plot_range=bounds)
            else:
                # Use default range
                self.update_plot(reset=True, plot_range=(-10, 10, None, None))

        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Invalid Function",
                                 f"Could not parse the function.\n\nError: {error_msg}\n\nTry using the Help button for syntax examples.")
            self.status_var.set(f"Error parsing function: {error_msg}")

    def update_plot(self, reset=False, plot_range=None):
        if not hasattr(self, 'expr') or self.expr is None:
            return

        self.ax.clear()

        # Get plot range from results, provided range, or use default
        if plot_range:
            if len(plot_range) == 2:
                start_plot, end_plot = plot_range
                y_min, y_max = None, None
            else:
                start_plot, end_plot, y_min, y_max = plot_range
        elif self.results and not reset:
            xl_min = min(row[1] for row in self.results)
            xu_max = max(row[3] for row in self.results)
            delta_x = self.results[0][2]

            # Expand range for better visualization
            start_plot = xl_min - delta_x * 3
            end_plot = xu_max + delta_x * 3

            # Ensure reasonable y-axis range
            y_values = []
            for row in self.results:
                y_values.append(row[4])  # f(xl)
                y_values.append(row[5])  # f(xu)

            if y_values:
                y_min = min(y_values)
                y_max = max(y_values)
                y_range = max(abs(y_max - y_min), 1)
                y_min = y_min - y_range * 0.2
                y_max = y_max + y_range * 0.2
        elif self.auto_bounds_var.get():
            # Use automatic bounds detection
            bounds = find_appropriate_bounds(self.expr, self.x_symbol)
            start_plot, end_plot, y_min, y_max = bounds
        else:
            # Default range if no results or reset requested
            start_plot = -10
            end_plot = 10
            y_min, y_max = None, None

        # Generate x values and calculate function
        xs = np.linspace(start_plot, end_plot, 1000)

        try:
            # Create a dictionary for constants
            constants = {
                'π': pi,
                'pi': pi,
                'e': E
            }

            # Substitute constants in the expression
            subs_dict = {symb: constants.get(str(symb), 0) for symb in self.expr.free_symbols if symb != self.x_symbol}

            f_lambdified = lambdify(self.x_symbol, self.expr.subs(subs_dict), modules=["numpy", "sympy"])
            ys = f_lambdified(xs)

            # Calculate y limits if not provided
            if y_min is None or y_max is None:
                # Filter out infinities and NaNs
                valid_ys = ys[~np.isinf(ys) & ~np.isnan(ys)]
                if len(valid_ys) > 0:
                    y_min = np.min(valid_ys)
                    y_max = np.max(valid_ys)
                    y_range = max(abs(y_max - y_min), 1)
                    y_min = y_min - y_range * 0.2
                    y_max = y_max + y_range * 0.2
                else:
                    y_min, y_max = -10, 10

            # Plot function curve
            self.ax.plot(xs, ys, color=self.plot_color, linewidth=2, label=f"f(x) = {self.func_entry.get()}")

            # Plot x and y axes
            self.ax.axhline(0, color="black", linewidth=0.7, alpha=0.7)
            self.ax.axvline(0, color="black", linewidth=0.7, alpha=0.7)

            # Plot points if enabled
            if self.show_points and self.results:
                for row in self.results:
                    xl, xu = row[1], row[3]
                    f_xl, f_xu = row[4], row[5]

                    # Plot points with annotations
                    self.ax.plot(xl, f_xl, 'ro', markersize=5)
                    self.ax.plot(xu, f_xu, 'bo', markersize=5)

                    # Add small annotations
                    self.ax.annotate(f'({xl:.2f}, {f_xl:.2f})',
                                     xy=(xl, f_xl),
                                     xytext=(5, 5),
                                     textcoords='offset points',
                                     fontsize=8)

                    self.ax.annotate(f'({xu:.2f}, {f_xu:.2f})',
                                     xy=(xu, f_xu),
                                     xytext=(5, 5),
                                     textcoords='offset points',
                                     fontsize=8)

            # Highlight root interval if found and enabled
            if self.show_root_area and self.results:
                for row in self.results:
                    if row[7] == "< 0" or row[7] == "= 0":  # Root exists in this interval or at a bound
                        xl, xu = row[1], row[3]
                        # Create a rectangle patch to highlight the root area
                        y_range = y_max - y_min
                        rect = Rectangle((xl, y_min), xu - xl, y_range,
                                         alpha=0.2, color='green')
                        self.ax.add_patch(rect)

                        # Add a vertical line at the approximate root
                        if row[7] == "= 0":
                            # If product is exactly 0, one of the bounds is a root
                            if row[4] == 0:  # f(xl) = 0
                                x_root = xl
                            else:  # f(xu) = 0
                                x_root = xu
                        else:
                            # Otherwise, estimate the root as the midpoint
                            x_root = (xl + xu) / 2

                        self.ax.axvline(x_root, color='green', linestyle='--', alpha=0.7)
                        self.ax.annotate(f'Root ≈ {x_root:.4f}',
                                         xy=(x_root, 0),
                                         xytext=(0, 30),
                                         textcoords='offset points',
                                         fontsize=10,
                                         ha='center',
                                         arrowprops=dict(arrowstyle='->'))
                        break

            # Set labels and grid
            self.ax.set_xlabel("x", fontsize=12, fontweight='bold')
            self.ax.set_ylabel("f(x)", fontsize=12, fontweight='bold')
            self.ax.set_title(f"Plot of f(x) = {self.func_entry.get()}", fontsize=14, fontweight='bold')

            # Set grid based on user preference
            self.ax.grid(self.grid_on, linestyle='--', alpha=0.7)

            # Set y limits if we have calculated them
            if y_min is not None and y_max is not None:
                self.ax.set_ylim(y_min, y_max)

            # Add legend
            self.ax.legend(loc='best', framealpha=0.7)

            # Make the plot look nicer
            self.fig.tight_layout()

            # Update status
            self.status_var.set(f"Function plotted successfully: {self.func_entry.get()}")

        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error plotting function: {str(e)}",
                         ha='center', va='center', transform=self.ax.transAxes,
                         color='red', fontsize=12)
            self.status_var.set(f"Error: {str(e)}")

        # Draw the updated plot
        self.canvas.draw()

    def compute(self):
        self.tree.delete(*self.tree.get_children())  # Clear previous results

        # Get the function string and preprocess it
        func_str = self.func_entry.get().strip()

        # Update status
        self.status_var.set(f"Processing function: {func_str}")

        # Preprocess the function string
        func_str = preprocess_function(func_str)

        # Insert multiplication signs where implied
        func_str = insert_multiplication_sign(func_str)

        try:
            x = symbols("x")
            self.x_symbol = x

            # Add special constants
            constants = {
                'π': pi,
                'pi': pi,
                'e': E
            }

            # Parse the expression with sympy
            self.expr = sympify(func_str, locals=constants)

            # Update status
            self.status_var.set(f"Function parsed successfully: {self.expr}")

        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Invalid Function",
                                 f"Could not parse the function.\n\nError: {error_msg}\n\nTry using the Help button for syntax examples.")
            self.status_var.set(f"Error parsing function: {error_msg}")
            return

        try:
            xl = float(self.start_entry.get())
            delta_x = float(self.increment_entry.get())
            max_iter = int(self.max_iter_entry.get())
            if delta_x <= 0:
                messagebox.showerror("Invalid Input", "Increment Δx must be > 0")
                self.status_var.set("Error: Increment Δx must be > 0")
                return
            if max_iter <= 0:
                messagebox.showerror("Invalid Input", "Max iterations must be > 0")
                self.status_var.set("Error: Max iterations must be > 0")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for x₀, Δx, and Max Iterations")
            self.status_var.set("Error: Invalid numerical inputs")
            return

        self.results = []
        root_found = False

        for i in range(1, max_iter + 1):
            xu = xl + delta_x
            try:
                f_xl = safe_eval(self.expr, xl)
                f_xu = safe_eval(self.expr, xu)

                # Check if we found an exact root
                if float(f_xl) == 0:
                    product_val = 0
                    remarks = "= 0"
                    root_found = True
                elif float(f_xu) == 0:
                    product_val = 0
                    remarks = "= 0"
                    root_found = True
                else:
                    product = f_xl * f_xu
                    product_val = float(product)
                    remarks = "> 0" if product_val > 0 else "< 0" if product_val < 0 else "= 0"
                    if product_val < 0:
                        root_found = True

            except Exception as e:
                error_msg = str(e)
                messagebox.showerror("Error",
                                     f"Error evaluating function at x = {xl} or x = {xu}.\n\nError: {error_msg}")
                self.status_var.set(f"Error evaluating function: {error_msg}")
                return

            self.results.append((i, xl, delta_x, xu, float(f_xl), float(f_xu), product_val, remarks))

            if root_found:
                break

            xl = xu

        # Add results to the table
        for row in self.results:
            formatted_row = (
                row[0],
                f"{row[1]:.6f}",
                f"{row[2]:.6f}",
                f"{row[3]:.6f}",
                f"{row[4]:.6f}",
                f"{row[5]:.6f}",
                f"{row[6]:.6f}",
                row[7]
            )
            self.tree.insert("", "end", values=formatted_row)

        # Always update the plot data
        if self.auto_bounds_var.get():
            # Use automatic bounds detection
            bounds = find_appropriate_bounds(self.expr, self.x_symbol)
            self.update_plot(plot_range=bounds)
        else:
            self.update_plot()

        # Switch to graph view to show the plot
        if self.view_mode != "graph":
            self.toggle_view()

        # Show a message if a root was found
        if root_found:
            for row in self.results:
                if row[7] == "< 0" or row[7] == "= 0":
                    xl, xu = row[1], row[3]

                    # Determine the root value
                    if row[7] == "= 0":
                        if row[4] == 0:  # f(xl) = 0
                            root_val = xl
                            root_msg = f"An exact root was found at x = {xl:.6f}"
                        else:  # f(xu) = 0
                            root_val = xu
                            root_msg = f"An exact root was found at x = {xu:.6f}"
                    else:
                        root_val = (xl + xu) / 2
                        root_msg = f"A root was found in the interval [{xl:.6f}, {xu:.6f}].\nApproximate root: {root_val:.6f}"

                    messagebox.showinfo("Root Found", root_msg)
                    self.status_var.set(f"Root found: {root_val:.6f}")
                    break
        else:
            # Just update the status bar, don't show an error message
            self.status_var.set("No root found in the specified range. Function plotted for visualization.")


# For testing the class directly
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    app = IncrementalSearchApp(root)
    app.protocol("WM_DELETE_WINDOW", root.destroy)  # Close the app properly
    root.mainloop()