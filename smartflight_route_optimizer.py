import json
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

console = Console()

# Load flight data from flights.json
def load_flight_data(filename="flights.json"):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        graph = {airport: {} for airport in data['airports']}
        airports = data['airports']
        for flight in data['flights']:
            graph[flight['from']][flight['to']] = {
                'cost': flight['cost'],
                'time': flight['time'],
                'distance': flight['distance']
            }
        return graph, airports
    except FileNotFoundError:
        console.print(f"[red]Error: {filename} not found.[/red]")
        return None, None

# A* Search: Fastest path by time
def a_star(graph, start, goal):
    def heuristic(node):
        return 1.0 if node != goal else 0.0  # Simple time-based heuristic
    
    open_list = [(heuristic(start), 0, start, [start], 0)]  # (f_score, g_score, current, path, total_distance)
    visited = set()
    g_scores = {start: 0}
    
    while open_list:
        _, g_score, current, path, total_distance = heapq.heappop(open_list)
        if current == goal:
            return path, g_score, total_distance
        if current not in visited:
            visited.add(current)
            for neighbor, attrs in graph[current].items():
                if neighbor not in visited:
                    new_g = g_score + attrs['time']
                    new_distance = total_distance + attrs['distance']
                    if neighbor not in g_scores or new_g < g_scores[neighbor]:
                        g_scores[neighbor] = new_g
                        f_score = new_g + heuristic(neighbor)
                        heapq.heappush(open_list, (f_score, new_g, neighbor, path + [neighbor], new_distance))
    return None, None, None

# Delay Simulation: Suggest alternative route if a flight is delayed
def delay_simulation(graph, original_path, delayed_flight, delay_hours):
    if not original_path:
        return None, None, None
    new_graph = graph.copy()
    for i in range(len(original_path) - 1):
        if (original_path[i], original_path[i+1]) == delayed_flight:
            new_graph[original_path[i]].pop(original_path[i+1], None)
    new_path, new_time, new_distance = a_star(new_graph, original_path[0], original_path[-1])
    return new_path, new_time, new_distance

# Visualization
def visualize_route(path, graph, title):
    G = nx.DiGraph()
    for start in graph:
        for end in graph[start]:
            G.add_edge(start, end)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
    plt.title(title)
    plt.show()

# Menu-based interface with rich formatting
def menu():
    graph, airports = load_flight_data()
    if not graph or not airports:
        console.print("[red]Exiting due to missing flight data.[/red]")
        return
    
    while True:
        console.print("\n[bold cyan]=== SmartFlight Route Optimizer (A* Algorithm) ===[/bold cyan]")
        
        # Display airports
        table = Table(title="Available Airports")
        table.add_column("No.", style="cyan", justify="center")
        table.add_column("Code", style="magenta")
        table.add_column("City", style="green")
        airport_names = {
            "ISB": "Islamabad", "KHI": "Karachi", "LHE": "Lahore", "PEW": "Peshawar",
            "DXB": "Dubai", "LHR": "London Heathrow", "JFK": "New York JFK", "NRT": "Tokyo Narita"
        }
        for i, airport in enumerate(airports, 1):
            table.add_row(str(i), airport, airport_names.get(airport, airport))
        console.print(table)
        
        # Get source airport
        src_idx = Prompt.ask(
            "\nSelect source airport (1-{})".format(len(airports)),
            console=console,
            choices=[str(i) for i in range(1, len(airports) + 1)],
            show_choices=False
        )
        try:
            src_idx = int(src_idx) - 1
            start_airport = airports[src_idx]
        except ValueError:
            console.print("[red]Invalid input. Try again.[/red]")
            continue
        
        # Get destination airport
        table = Table(title="Available Destination Airports")
        table.add_column("No.", style="cyan", justify="center")
        table.add_column("Code", style="magenta")
        table.add_column("City", style="green")
        for i, airport in enumerate(airports, 1):
            table.add_row(str(i), airport, airport_names.get(airport, airport))
        console.print(table)
        
        dest_idx = Prompt.ask(
            "Select destination airport (1-{})".format(len(airports)),
            console=console,
            choices=[str(i) for i in range(1, len(airports) + 1)],
            show_choices=False
        )
        try:
            dest_idx = int(dest_idx) - 1
            goal_airport = airports[dest_idx]
            if goal_airport == start_airport:
                console.print("[red]Source and destination cannot be the same.[/red]")
                continue
        except ValueError:
            console.print("[red]Invalid input. Try again.[/red]")
            continue
        
        # Run A* algorithm
        console.print("\n[bold yellow]Results:[/bold yellow]")
        path, time, distance = a_star(graph, start_airport, goal_airport)
        if path:
            console.print(f"[green]A* (min time): {' → '.join(path)}[/green]")
            console.print(f"[cyan]Time: {time} hours[/cyan]")
            console.print(f"[cyan]Distance: {distance} km[/cyan]")
            if len(path) > 1:
                delayed_flight = (path[0], path[1])
                alt_path, alt_time, alt_distance = delay_simulation(graph, path, delayed_flight, 2.0)
                if alt_path:
                    console.print(f"\n[green]A* Delay Simulation (delayed {delayed_flight}): {' → '.join(alt_path)}[/green]")
                    console.print(f"[cyan]Alternative time: {alt_time} hours[/cyan]")
                    console.print(f"[cyan]Alternative distance: {alt_distance} km[/cyan]")
                    visualize_route(alt_path, graph, f"A* Alternative Path (Delay): {start_airport} to {goal_airport}")
                else:
                    console.print(f"[red]A* Delay Simulation (delayed {delayed_flight}): No alternative path found.[/red]")
            visualize_route(path, graph, f"A* Path: {start_airport} to {goal_airport}")
        else:
            console.print("[red]No path found with A*.[/red]")
        
        # Ask to continue
        cont = Prompt.ask(
            "\nDo you want to try another route? (y/n)",
            console=console,
            choices=["y", "n"],
            default="n"
        )
        if cont.lower() != 'y':
            console.print("[bold cyan]Thank you for using SmartFlight Route Optimizer![/bold cyan]")
            break

if __name__ == "__main__":
    menu()

