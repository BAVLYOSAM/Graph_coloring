import tkinter as tk
from tkinter import ttk, messagebox
import random
import timeit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from tkinter import font as tkfont
# ************GUI************


class GraphColoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Graph Coloring Solver")
        self.root.geometry("1300x800")
        self.root.configure(bg='#f5f5f5')

        # Custom fonts
        self.title_font = tkfont.Font(
            family="Helvetica", size=14, weight="bold")
        self.subtitle_font = tkfont.Font(
            family="Helvetica", size=11, weight="bold")
        self.normal_font = tkfont.Font(family="Helvetica", size=10)

        # Configure style
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f5f5f5')
        self.style.configure(
            'TLabel', background='#f5f5f5', font=self.normal_font)
        self.style.configure('TButton', font=self.normal_font, padding=5)
        self.style.configure('Header.TLabel', font=self.title_font)
        self.style.configure('Subheader.TLabel', font=self.subtitle_font)
        self.style.configure('Result.TLabel', font=("Helvetica", 10, "bold"))
        self.style.configure('TEntry', padding=5)
        self.style.configure('TCombobox', padding=5)

        # Main frames
        self.control_frame = ttk.Frame(root, padding="15", style='TFrame')
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.visualization_frame = ttk.Frame(
            root, padding="15", style='TFrame')
        self.visualization_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Control panel widgets
        self.create_control_panel()

        # Visualization components
        self.figure = plt.figure(figsize=(8, 6), dpi=100, facecolor='#f5f5f5')
        self.canvas = FigureCanvasTkAgg(
            self.figure, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Results display
        self.results_frame = ttk.LabelFrame(
            self.control_frame, text="Results", padding=10)
        self.results_frame.pack(fill=tk.X, pady=10)

        self.results_text = tk.Text(self.results_frame, height=15, width=40, wrap=tk.WORD,
                                    font=self.normal_font, bg='white', padx=10, pady=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Initialize graph
        self.graph = {}
        self.num_vertices = 0
        self.edges = []
        self.current_coloring = []

    def create_control_panel(self):
        # Header
        header = ttk.Label(self.control_frame,
                           text="Graph Coloring Solver", style='Header.TLabel')
        header.pack(pady=(0, 15))

        # Graph setup section
        setup_frame = ttk.LabelFrame(
            self.control_frame, text="Graph Setup", padding=10)
        setup_frame.pack(fill=tk.X, pady=5)

        ttk.Label(setup_frame, text="Number of Nodes:").pack(anchor=tk.W)
        self.node_entry = ttk.Entry(setup_frame)
        self.node_entry.pack(fill=tk.X, pady=2)

        ttk.Label(setup_frame, text="Number of Edges:").pack(anchor=tk.W)
        self.edge_entry = ttk.Entry(setup_frame)
        self.edge_entry.pack(fill=tk.X, pady=2)

        ttk.Button(setup_frame, text="Initialize Graph",
                   command=self.initialize_graph).pack(fill=tk.X, pady=10)

        # Edge input section
        self.edge_input_frame = ttk.LabelFrame(
            self.control_frame, text="Add Edges", padding=10)
        self.edge_input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.edge_input_frame, text="From Node:").pack(anchor=tk.W)
        self.from_node = ttk.Combobox(self.edge_input_frame, state="readonly")
        self.from_node.pack(fill=tk.X, pady=2)

        ttk.Label(self.edge_input_frame, text="To Node:").pack(anchor=tk.W)
        self.to_node = ttk.Combobox(self.edge_input_frame, state="readonly")
        self.to_node.pack(fill=tk.X, pady=2)

        ttk.Button(self.edge_input_frame, text="Add Edge",
                   command=self.add_edge).pack(fill=tk.X, pady=5)

        # Algorithm parameters
        algo_frame = ttk.LabelFrame(
            self.control_frame, text="Algorithm Parameters", padding=10)
        algo_frame.pack(fill=tk.X, pady=5)

        ttk.Label(algo_frame, text="Maximum Colors to Test:").pack(anchor=tk.W)
        self.max_colors_entry = ttk.Entry(algo_frame)
        self.max_colors_entry.insert(0, "5")
        self.max_colors_entry.pack(fill=tk.X, pady=2)

        ttk.Label(algo_frame, text="Population Size:").pack(anchor=tk.W)
        self.pop_size_entry = ttk.Entry(algo_frame)
        self.pop_size_entry.insert(0, "20")
        self.pop_size_entry.pack(fill=tk.X, pady=2)

        ttk.Label(algo_frame, text="Generations:").pack(anchor=tk.W)
        self.generations_entry = ttk.Entry(algo_frame)
        self.generations_entry.insert(0, "100")
        self.generations_entry.pack(fill=tk.X, pady=2)

        ttk.Label(algo_frame, text="Mutation Rate:").pack(anchor=tk.W)
        self.mutation_entry = ttk.Entry(algo_frame)
        self.mutation_entry.insert(0, "0.01")
        self.mutation_entry.pack(fill=tk.X, pady=2)

        # Run buttons
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Find Chromatic Number",
                   command=self.find_chromatic_number).pack(side=tk.LEFT, expand=True, padx=2)

        ttk.Button(button_frame, text="Run Genetic Algorithm",
                   command=lambda: self.run_algorithm("genetic")).pack(side=tk.LEFT, expand=True, padx=2)

        ttk.Button(button_frame, text="Run Backtracking",
                   command=lambda: self.run_algorithm("backtracking")).pack(side=tk.LEFT, expand=True, padx=2)

    def initialize_graph(self):
        try:
            self.num_vertices = int(self.node_entry.get())
            num_edges = int(self.edge_entry.get())

            if self.num_vertices < 1:
                raise ValueError("Number of nodes must be at least 1")
            if num_edges < 0 or num_edges > self.num_vertices * (self.num_vertices - 1) // 2:
                raise ValueError("Invalid number of edges")

            # Initialize empty graph
            self.graph = {i: [] for i in range(1, self.num_vertices + 1)}
            self.edges = []

            # Update node selection dropdowns
            nodes = list(self.graph.keys())
            self.from_node['values'] = nodes
            self.to_node['values'] = nodes

            messagebox.showinfo(
                "Success", f"Graph with {self.num_vertices} nodes initialized.\nNow add {num_edges} edges.")
            self.display_graph()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def add_edge(self):
        try:
            from_node = int(self.from_node.get())
            to_node = int(self.to_node.get())

            if from_node == to_node:
                raise ValueError("Cannot add edge from a node to itself")
            if to_node in self.graph[from_node]:
                raise ValueError("Edge already exists")

            self.graph[from_node].append(to_node)
            self.graph[to_node].append(from_node)
            self.edges.append((from_node, to_node))

            self.display_graph()
            self.results_text.insert(
                tk.END, f"Added edge: {from_node} - {to_node}\n")
            self.results_text.see(tk.END)

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid edge: {str(e)}")

    def display_graph(self, coloring=None):
        if not self.graph:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#f5f5f5')

        # Create networkx graph
        G = nx.Graph()
        for node in self.graph:
            G.add_node(node)
        for edge in self.edges:
            G.add_edge(edge[0], edge[1])

        # Determine node colors
        node_colors = []
        if coloring and len(coloring) == self.num_vertices:
            color_map = plt.cm.get_cmap('tab20', max(coloring) + 1)
            node_colors = [color_map(c) for c in coloring]
        else:
            node_colors = ['#4682b4'] * self.num_vertices  # Steel blue

        # Draw the graph
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700,
                               node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=ax, width=2,
                               alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(
            G, pos, ax=ax, font_size=12, font_weight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()

    def run_algorithm(self, algorithm):
        if not self.graph or not self.edges:
            messagebox.showerror(
                "Error", "Please initialize the graph and add edges first.")
            return

        try:
            max_colors = int(self.max_colors_entry.get())

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(
                tk.END, f"Running {algorithm} algorithm...\n")
            self.results_text.see(tk.END)
            self.root.update()

            if algorithm == "genetic":
                pop_size = int(self.pop_size_entry.get())
                generations = int(self.generations_entry.get())
                mutation_rate = float(self.mutation_entry.get())

                # Use timeit for accurate timing
                start_time = timeit.default_timer()
                solution, conflicts = self.genetic_algorithm(
                    self.graph, max_colors, pop_size, generations, mutation_rate)
                exec_time = timeit.default_timer() - start_time

                self.display_results(algorithm, solution, conflicts, exec_time,
                                     pop_size, generations, mutation_rate)
                self.display_graph(solution)

            elif algorithm == "backtracking":
                # Use timeit for accurate timing
                start_time = timeit.default_timer()
                solution = self.backtracking(self.graph, max_colors)
                exec_time = timeit.default_timer() - start_time

                conflicts = self.calculate_conflicts(solution, self.graph)
                self.display_results(algorithm, solution, conflicts, exec_time)
                self.display_graph(solution)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run algorithm: {str(e)}")

    def find_chromatic_number(self):
        if not self.graph or not self.edges:
            messagebox.showerror(
                "Error", "Please initialize the graph and add edges first.")
            return

        try:
            max_colors = int(self.max_colors_entry.get())

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Finding chromatic number...\n")
            self.results_text.see(tk.END)
            self.root.update()

            # Use timeit for accurate timing
            start_time = timeit.default_timer()
            chromatic_num, coloring = self.find_chromatic_number_impl(
                self.graph, max_colors)
            exec_time = timeit.default_timer() - start_time

            if chromatic_num:
                conflicts = self.calculate_conflicts(coloring, self.graph)
                self.display_results("Chromatic Number",
                                     coloring, conflicts, exec_time)
                self.display_graph(coloring)
                self.results_text.insert(
                    tk.END, f"\nChromatic Number: {chromatic_num}\n", 'success')
                self.results_text.tag_config(
                    'success', foreground='green', font=("Helvetica", 11, "bold"))
            else:
                self.results_text.insert(
                    tk.END, f"No valid coloring found with up to {max_colors} colors\n", 'warning')
                self.results_text.tag_config(
                    'warning', foreground='red', font=("Helvetica", 11, "bold"))

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to find chromatic number: {str(e)}")

    def display_results(self, algorithm, solution, conflicts, exec_time, pop_size=None, generations=None, mutation_rate=None):
        self.results_text.insert(
            tk.END, f"\n--- {algorithm} Results ---\n", 'subheader')
        self.results_text.tag_config('subheader', font=self.subtitle_font)

        n = len(self.graph)
        m = int(self.max_colors_entry.get())

        # Add detailed time complexity information with actual parameters
        if algorithm == "genetic":
            p = pop_size
            g = generations
            time_complexity = f"O(g × p × n²) = O({g} × {p} × {n}²) = O({g * p * n * n}) operations"
            self.results_text.insert(
                tk.END, f"Parameters:\n- Population: {p}\n- Generations: {g}\n- Mutation Rate: {mutation_rate}\n")

        elif algorithm == "backtracking":
            time_complexity = f"O(m^n) = O({m}^{n}) = O({m**n}) operations (worst case)"

        elif algorithm == "Chromatic Number":
            time_complexity = f"O(m × m^n) = O({m} × {m}^{n}) = O({m * (m**n)}) operations (worst case)"

        self.results_text.insert(
            tk.END, f"Time Complexity: {time_complexity}\n")

        # Format execution time to show more precision
        if exec_time < 0.0001:
            time_str = f"{exec_time:.10f} seconds"
        elif exec_time < 1:
            time_str = f"{exec_time:.6f} seconds"
        else:
            time_str = f"{exec_time:.4f} seconds"

        self.results_text.insert(tk.END, f"Execution Time: {time_str}\n")
        self.results_text.insert(tk.END, f"Number of Conflicts: {conflicts}\n")
        self.results_text.insert(tk.END, f"Coloring: {solution}\n")

        if conflicts == 0:
            self.results_text.insert(
                tk.END, "VALID COLORING FOUND!\n", 'success')
            self.results_text.tag_config(
                'success', foreground='green', font=("Helvetica", 10, "bold"))
        else:
            self.results_text.insert(
                tk.END, "Conflict detected in coloring\n", 'warning')
            self.results_text.tag_config(
                'warning', foreground='red', font=("Helvetica", 10, "bold"))

        self.results_text.see(tk.END)

    # ************genetic algorithms ************

    def genetic_algorithm(self, graph, numColors, populationSize=20, generations=100, mutationRate=0.01):
        numVertices = len(graph)
        population = [self.generateIndividual(
            numVertices, numColors) for _ in range(populationSize)]
        best_solution = None
        best_score = float('-inf')

        for _ in range(generations):
            scores = [self.fitness(ind, graph) for ind in population]
            current_best = max(scores)
            if current_best > best_score:
                best_score = current_best
                best_solution = population[scores.index(current_best)]
                if best_score == 0:  # Found perfect solution
                    break

            new_population = []
            for _ in range(populationSize):
                parent1 = self.selection(population, scores)
                parent2 = self.selection(population, scores)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child, numColors, mutationRate)
                new_population.append(child)
            population = new_population

        return best_solution, -best_score

    def generateIndividual(self, numVertices, numColors):
        return [random.randint(1, numColors) for _ in range(numVertices)]

    def fitness(self, individual, graph):
        conflict = 0
        for node in graph:
            for neighbour in graph[node]:
                if individual[node-1] == individual[neighbour-1]:
                    conflict += 1
        return -conflict  # We want to minimize conflicts

    def selection(self, population, scores):
        min_score = min(scores)
        # Shift to positive values
        shifted = [s - min_score + 1 for s in scores]
        total = sum(shifted)
        probs = [s / total for s in shifted]
        return population[random.choices(range(len(population)), weights=probs)[0]]

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1)-2)  # Avoid endpoints
        return parent1[:point] + parent2[point:]

    def mutation(self, individual, numColors, mutationRate):
        new = individual[:]
        for i in range(len(new)):
            if random.random() < mutationRate:
                new[i] = random.randint(1, numColors)
        return new

    # ************backtracking************

    def backtracking(self, graph, numColors):
        colors = {}

        def solve(node):
            if node > len(graph):
                return True
            for c in range(1, numColors + 1):
                if self.checkColor(graph, colors, node, c):
                    colors[node] = c
                    if solve(node + 1):
                        return True
                    colors[node] = 0
            return False

        solve(1)  # Start with node 1
        return [colors.get(i, 0) for i in range(1, len(graph)+1)]

    def checkColor(self, graph, colors, node, color):
        for neighbor in graph[node]:
            if colors.get(neighbor) == color:  # c
                return False
        return True

    # ************conflicts************

    def calculate_conflicts(self, coloring, graph):
        conflicts = 0
        for node in graph:
            for neighbor in graph[node]:
                if coloring[node-1] == coloring[neighbor-1]:
                    conflicts += 1
        return conflicts // 2  # Each conflict counted twice

    def find_chromatic_number_impl(self, graph, max_colors=10):
        for c in range(1, max_colors + 1):
            coloring = self.backtracking(graph, c)
            if 0 not in coloring:  # Valid coloring found
                return c, coloring
        return None, None


if __name__ == "__main__":
    root = tk.Tk()
    app = GraphColoringApp(root)
    root.mainloop()
