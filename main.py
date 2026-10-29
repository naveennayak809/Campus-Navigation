import sqlite3
from datetime import datetime
from pathlib import Path
import heapq
import matplotlib.pyplot as plt
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = DATA_DIR / "route_history.db"
MAP_PATH = OUTPUT_DIR / "campus_map.png"

# Illustrative sample distances in metres; replace with measured campus data.
EDGES = [
    ("Main Gate", "Admin Block", 250),
    ("Main Gate", "Canteen", 180),
    ("Admin Block", "Library", 220),
    ("Admin Block", "Academic Block", 160),
    ("Canteen", "Library", 270),
    ("Canteen", "Hostel", 350),
    ("Library", "Academic Block", 120),
    ("Library", "Auditorium", 200),
    ("Academic Block", "Auditorium", 140),
    ("Academic Block", "Hostel", 300),
    ("Auditorium", "Sports Ground", 260),
    ("Hostel", "Sports Ground", 210),
]
LOCATIONS = sorted({p for a, b, _ in EDGES for p in (a, b)})

def build_graph(blocked_edges=None):
    graph = nx.Graph()
    graph.add_nodes_from(LOCATIONS)
    blocked = {frozenset(edge) for edge in (blocked_edges or [])}
    for start, end, distance in EDGES:
        if frozenset((start, end)) not in blocked:
            graph.add_edge(start, end, weight=distance)
    return graph

def dijkstra(graph, source, destination):
    distances = {node: float("inf") for node in graph.nodes}
    previous = {}
    distances[source] = 0
    queue = [(0, source)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance != distances[current]:
            continue
        if current == destination:
            break
        for neighbor, data in graph[current].items():
            candidate = current_distance + data["weight"]
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    if distances[destination] == float("inf"):
        return None, float("inf")
    path, current = [], destination
    while current != source:
        path.append(current)
        current = previous[current]
    path.append(source)
    path.reverse()
    return path, distances[destination]

def setup_database():
    DATA_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS route_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL, destination TEXT NOT NULL, route TEXT NOT NULL,
            distance_m REAL NOT NULL, walking_minutes REAL NOT NULL, searched_at TEXT NOT NULL
        )""")

def save_search(source, destination, path, distance, minutes):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """INSERT INTO route_history
            (source, destination, route, distance_m, walking_minutes, searched_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (source, destination, " -> ".join(path), distance, minutes,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

def show_history():
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            "SELECT source, destination, route, distance_m, walking_minutes, searched_at "
            "FROM route_history ORDER BY id DESC LIMIT 20"
        ).fetchall()
    if not rows:
        print("\nNo route searches saved yet.")
        return
    print("\n--- Recent Route Searches ---")
    for source, destination, route, distance, minutes, timestamp in rows:
        print(f"\n{source} -> {destination}\nRoute: {route}")
        print(f"Distance: {distance:.0f} m | Estimated time: {minutes:.1f} min")
        print(f"Searched at: {timestamp}")

def draw_map(graph, path=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    positions = nx.spring_layout(graph, seed=17, weight="weight")
    plt.figure(figsize=(12, 7))
    nx.draw_networkx_nodes(graph, positions, node_size=1800)
    nx.draw_networkx_labels(graph, positions, font_size=8)
    nx.draw_networkx_edges(graph, positions, width=1.6, edge_color="gray")
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=nx.get_edge_attributes(graph, "weight"), font_size=7)
    if path and len(path) > 1:
        nx.draw_networkx_nodes(graph, positions, nodelist=path, node_size=1900, node_color="orange")
        nx.draw_networkx_edges(graph, positions, edgelist=list(zip(path, path[1:])), width=4, edge_color="red")
    plt.title("Smart Campus Navigation — sample distances in metres")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(MAP_PATH, dpi=160)
    plt.show()
    print(f"Map image saved to: {MAP_PATH}")

def choose_location(prompt):
    while True:
        value = input(prompt).strip()
        for location in LOCATIONS:
            if location.lower() == value.lower():
                return location
        print("Location not found. Available locations:")
        print(", ".join(LOCATIONS))

def main():
    setup_database()
    graph = build_graph()
    while True:
        print("\n========== SMART CAMPUS NAVIGATION ==========")
        print("1. Find shortest route")
        print("2. Find route while avoiding a blocked path")
        print("3. Display campus map")
        print("4. Show recent route history")
        print("5. List campus locations")
        print("0. Exit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            source = choose_location("Starting location: ")
            destination = choose_location("Destination: ")
            if source == destination:
                print("You are already at the destination.")
                continue
            path, distance = dijkstra(graph, source, destination)
            if path is None:
                print("No route is available.")
                continue
            minutes = distance / 80  # Assumes walking speed of 80 metres/minute.
            print("\nShortest route:", " -> ".join(path))
            print(f"Total distance: {distance:.0f} metres")
            print(f"Estimated walking time: {minutes:.1f} minutes")
            save_search(source, destination, path, distance, minutes)
            draw_map(graph, path)

        elif choice == "2":
            print("\nAvailable connections:")
            for start, end, distance in EDGES:
                print(f"{start} <-> {end}: {distance} m")
            blocked_start = choose_location("First location of blocked path: ")
            blocked_end = choose_location("Second location of blocked path: ")
            if not graph.has_edge(blocked_start, blocked_end):
                print("Those locations do not have a direct connection.")
                continue
            updated_graph = build_graph([(blocked_start, blocked_end)])
            source = choose_location("Starting location: ")
            destination = choose_location("Destination: ")
            path, distance = dijkstra(updated_graph, source, destination)
            if path is None:
                print("No route is available while that path is blocked.")
                continue
            minutes = distance / 80
            print("\nAlternative route:", " -> ".join(path))
            print(f"Total distance: {distance:.0f} metres")
            print(f"Estimated walking time: {minutes:.1f} minutes")
            save_search(source, destination, path, distance, minutes)
            draw_map(updated_graph, path)

        elif choice == "3":
            draw_map(graph)
        elif choice == "4":
            show_history()
        elif choice == "5":
            print("\nCampus locations:")
            for location in LOCATIONS:
                print("-", location)
        elif choice == "0":
            print("Thank you for using Smart Campus Navigation.")
            break
        else:
            print("Invalid choice. Please select an option from the menu.")

if __name__ == "__main__":
    main()
