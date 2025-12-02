"""
Interactive Python demo: Uninformed (Blind) Search Strategies

Includes:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Depth-Limited Search (DLS)
- Iterative Deepening Search (IDS)
- Uniform-Cost Search (UCS)
- Bidirectional Search

Run the file in a terminal (Windows: `python untitled.py` or from VS Code).
The program is interactive: choose a sample graph or define a simple one,
pick a start and goal node, then pick a search strategy. The demo will
explain the algorithm, show step-by-step frontier/expanded nodes, and
print the final path and stats.

Notes:
- Graphs are small for clarity. UCS expects non-negative edge costs.
- Bidirectional search assumes an undirected graph (same adjacency both ways).
"""

import heapq
from collections import deque, defaultdict
import sys

# Simple graph class
class Graph:
    def __init__(self, directed=False):
        self.adj = defaultdict(list)  # node -> list of (neighbor, cost)
        self.directed = directed

    def add_edge(self, u, v, cost=1):
        self.adj[u].append((v, cost))
        if not self.directed:
            self.adj[v].append((u, cost))

    def neighbors(self, u):
        return self.adj.get(u, [])

    def nodes(self):
        return list(self.adj.keys())


# Utility: reconstruct path from parent map
def reconstruct_path(parent, start, goal):
    if goal not in parent and start != goal:
        return None
    path = []
    cur = goal
    while cur != start and cur in parent:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


# Breadth-First Search (graph search)
def bfs(graph, start, goal, explain=True):
    if explain:
        print("\n--- Breadth-First Search (BFS) ---")
        print("BFS explores the shallowest nodes first using a FIFO queue.")
    queue = deque([start])
    parent = {}
    expanded = set()
    steps = 0

    while queue:
        frontier = list(queue)
        if explain:
            print(f"Frontier: {frontier}")
        node = queue.popleft()
        if explain:
            print(f"Expanding: {node}")
        steps += 1
        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return {"path": path, "expanded": expanded, "steps": steps}
        expanded.add(node)
        for (nbr, _) in graph.neighbors(node):
            if nbr not in parent and nbr not in expanded and nbr not in queue and nbr != start:
                parent[nbr] = node
                queue.append(nbr)
        if explain:
            print(f"Queue after expansion: {list(queue)}\n")
    return {"path": None, "expanded": expanded, "steps": steps}


# Depth-First Search (graph search using stack)
def dfs(graph, start, goal, explain=True, depth_limit=None):
    if explain:
        if depth_limit is None:
            print("\n--- Depth-First Search (DFS) ---")
            print("DFS explores as deep as possible along each branch before backtracking.")
        else:
            print(f"\n--- Depth-Limited DFS (limit={depth_limit}) ---")
            print("DFS with a maximum depth; nodes deeper than limit won't be expanded.")

    stack = [(start, 0)]
    parent = {}
    expanded = set()
    steps = 0
    visited_depth = {start: 0}

    while stack:
        node, d = stack.pop()
        if explain:
            print(f"Popped: {node} (depth {d})")
        steps += 1
        if node == goal:
            path = reconstruct_path(parent, start, goal)
            return {"path": path, "expanded": expanded, "steps": steps}
        if node not in expanded:
            expanded.add(node)
            if depth_limit is None or d < depth_limit:
                # push neighbors in reverse to preserve left-to-right order if needed
                for (nbr, _) in reversed(graph.neighbors(node)):
                    nd = d + 1
                    if nbr not in visited_depth or nd < visited_depth[nbr]:
                        parent[nbr] = node
                        visited_depth[nbr] = nd
                        stack.append((nbr, nd))
        if explain:
            print(f"Stack now: {[n for n,d in stack]}\n")
    return {"path": None, "expanded": expanded, "steps": steps}


# Iterative Deepening Search (IDS)
def ids(graph, start, goal, max_depth=20, explain=True):
    if explain:
        print("\n--- Iterative Deepening Search (IDS) ---")
        print("IDS does DFS with increasing depth limits until the goal is found.")
    for limit in range(max_depth + 1):
        if explain:
            print(f"\nTrying depth limit = {limit}")
        result = dfs(graph, start, goal, explain=explain, depth_limit=limit)
        if result["path"] is not None:
            result["limit_used"] = limit
            return result
    return {"path": None, "expanded": set(), "steps": 0}


# Uniform-Cost Search (UCS)
def ucs(graph, start, goal, explain=True):
    if explain:
        print("\n--- Uniform-Cost Search (UCS) ---")
        print("UCS expands the node with the lowest path cost so far (Dijkstra).")
    frontier = []
    heapq.heappush(frontier, (0, start))
    parent = {}
    cost_so_far = {start: 0}
    expanded = set()
    steps = 0

    while frontier:
        cost, node = heapq.heappop(frontier)
        if explain:
            frontier_nodes = [(c, n) for (c, n) in frontier]
            print(f"Frontier: {[(c, n) for (c,n) in frontier_nodes]}; popping {(cost, node)}")
        if node == goal:
            if explain:
                print(f"Reached goal {goal} with cost {cost}")
            path = reconstruct_path(parent, start, goal)
            return {"path": path, "cost": cost, "expanded": expanded, "steps": steps}
        if node in expanded:
            continue
        expanded.add(node)
        steps += 1
        for (nbr, edge_cost) in graph.neighbors(node):
            new_cost = cost_so_far[node] + edge_cost
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                parent[nbr] = node
                heapq.heappush(frontier, (new_cost, nbr))
        if explain:
            print(f"Expanded: {node}; cost_so_far: {cost_so_far}\n")
    return {"path": None, "cost": None, "expanded": expanded, "steps": steps}


# Bidirectional search (meets in the middle). Assumes undirected graph.
def bidirectional_search(graph, start, goal, explain=True):
    if explain:
        print("\n--- Bidirectional Search ---")
        print("Runs two simultaneous BFS from start and goal and meets in the middle.")
    if start == goal:
        return {"path": [start], "expanded": set(), "steps": 0}

    # frontier and parent maps for both searches
    frontier_f = deque([start])
    frontier_b = deque([goal])
    parent_f = {}
    parent_b = {}
    expanded_f = set()
    expanded_b = set()
    steps = 0
    visited_f = {start}
    visited_b = {goal}

    while frontier_f and frontier_b:
        # expand forward
        node_f = frontier_f.popleft()
        if explain:
            print(f"Forward expanding: {node_f}; frontier_f: {list(frontier_f)}")
        steps += 1
        expanded_f.add(node_f)
        for (nbr, _) in graph.neighbors(node_f):
            if nbr not in visited_f:
                visited_f.add(nbr)
                parent_f[nbr] = node_f
                frontier_f.append(nbr)
                if nbr in visited_b:
                    # meeting point
                    meeting = nbr
                    path_f = reconstruct_path(parent_f, start, meeting)
                    path_b = reconstruct_path(parent_b, goal, meeting)
                    if path_b is None:
                        path_b = [meeting]
                    else:
                        path_b = path_b[::-1]
                    full_path = path_f + path_b[1:]  # avoid double meeting node
                    return {"path": full_path, "expanded": expanded_f.union(expanded_b), "steps": steps}
        # expand backward
        node_b = frontier_b.popleft()
        if explain:
            print(f"Backward expanding: {node_b}; frontier_b: {list(frontier_b)}")
        steps += 1
        expanded_b.add(node_b)
        for (nbr, _) in graph.neighbors(node_b):
            if nbr not in visited_b:
                visited_b.add(nbr)
                parent_b[nbr] = node_b
                frontier_b.append(nbr)
                if nbr in visited_f:
                    meeting = nbr
                    path_f = reconstruct_path(parent_f, start, meeting)
                    path_b = reconstruct_path(parent_b, goal, meeting)
                    if path_b is None:
                        path_b = [meeting]
                    else:
                        path_b = path_b[::-1]
                    full_path = path_f + path_b[1:]
                    return {"path": full_path, "expanded": expanded_f.union(expanded_b), "steps": steps}
    return {"path": None, "expanded": expanded_f.union(expanded_b), "steps": steps}


# Sample graphs for demo
def sample_graphs():
    g1 = Graph(directed=False)
    edges = [
        ("A","B"),("A","C"),("B","D"),("C","E"),("D","F"),("E","F"),("F","G")
    ]
    for u,v in edges:
        g1.add_edge(u,v)
    g2 = Graph(directed=False)
    # weighted graph for UCS demo
    g2.add_edge("S","A",2)
    g2.add_edge("S","B",5)
    g2.add_edge("A","C",2)
    g2.add_edge("B","C",1)
    g2.add_edge("C","G",3)
    # graph for bidirectional
    g3 = Graph(directed=False)
    path = ["S","A","B","C","D","G"]
    for i in range(len(path)-1):
        g3.add_edge(path[i], path[i+1])
    return {"Small": g1, "Weighted": g2, "LongLine": g3}


def print_graph(graph):
    print("\nGraph adjacency list (node: [(neighbor,cost), ...]):")
    for n in sorted(graph.adj.keys()):
        print(f"  {n}: {graph.adj[n]}")


def choose_graph():
    graphs = sample_graphs()
    print("Available sample graphs:")
    for i, name in enumerate(graphs.keys(), start=1):
        print(f"  {i}. {name}")
    print("  0. Define a tiny custom graph (comma-separated edges like A-B,A-C)")
    choice = input("Pick a graph (number): ").strip()
    if choice == "0":
        g = Graph(directed=False)
        edge_str = input("Enter edges (like A-B:1,A-C:2 or A-B,B-C for unit weights): ").strip()
        if not edge_str:
            print("No edges provided. Using empty graph.")
            return g
        parts = [p.strip() for p in edge_str.split(",") if p.strip()]
        for p in parts:
            if ":" in p:
                e, c = p.split(":")
                u,v = e.split("-")
                g.add_edge(u.strip(), v.strip(), float(c))
            else:
                u,v = p.split("-")
                g.add_edge(u.strip(), v.strip(), 1)
        return g
    else:
        try:
            idx = int(choice) - 1
            key = list(graphs.keys())[idx]
            return graphs[key]
        except Exception:
            print("Invalid choice; using Small graph.")
            return graphs["Small"]


def main():
    print("Uninformed Search Strategies Interactive Demo")
    graph = choose_graph()
    print_graph(graph)
    start = input("\nStart node: ").strip()
    goal = input("Goal node: ").strip()
    if start == "" or goal == "":
        print("Start and goal must be provided.")
        return

    print("\nChoose a search strategy:")
    print("  1. Breadth-First Search (BFS)")
    print("  2. Depth-First Search (DFS)")
    print("  3. Depth-Limited Search (DLS)")
    print("  4. Iterative Deepening Search (IDS)")
    print("  5. Uniform-Cost Search (UCS)")
    print("  6. Bidirectional Search")
    choice = input("Strategy number: ").strip()

    if choice == "1":
        res = bfs(graph, start, goal, explain=True)
        print("\nResult:", res["path"])
    elif choice == "2":
        res = dfs(graph, start, goal, explain=True, depth_limit=None)
        print("\nResult:", res["path"])
    elif choice == "3":
        lim = input("Enter depth limit (integer): ").strip()
        try:
            lim = int(lim)
        except:
            lim = 3
            print("Invalid limit; using 3.")
        res = dfs(graph, start, goal, explain=True, depth_limit=lim)
        print("\nResult:", res["path"])
    elif choice == "4":
        maxd = input("Enter max depth for IDS (integer): ").strip()
        try:
            maxd = int(maxd)
        except:
            maxd = 10
        res = ids(graph, start, goal, max_depth=maxd, explain=True)
        print("\nResult:", res.get("path"), "; limit used:", res.get("limit_used"))
    elif choice == "5":
        res = ucs(graph, start, goal, explain=True)
        print("\nResult path:", res["path"], " cost:", res.get("cost"))
    elif choice == "6":
        res = bidirectional_search(graph, start, goal, explain=True)
        print("\nResult:", res["path"])
    else:
        print("Unknown choice.")

    print("\nExpanded nodes:", sorted(list(res.get("expanded", []))))
    print("Steps (number of expansions or pops):", res.get("steps"))
    print("\nDemo finished.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")