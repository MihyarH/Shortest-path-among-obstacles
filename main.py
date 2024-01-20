import tkinter as tk
import heapq
import random
import threading
from collections import deque

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_obstacle = False
        self.parent = None
        self.g_score = float('inf')
        self.h_score = 0

    def __lt__(self, other):
        return False

def heuristic(node, goal):
    return abs(node.x - goal.x) + abs(node.y - goal.y)

def get_neighbors(node, grid):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = node.x + dx, node.y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and not grid[nx][ny].is_obstacle:
            neighbors.append(grid[nx][ny])
    return neighbors

def astar(start, goal, grid):
    open_set = []
    count = 0
    heapq.heappush(open_set, (0, count, start))
    start.g_score = 0

    while open_set:
        current_cost, _, current_node = heapq.heappop(open_set)

        if current_node == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        for neighbor in get_neighbors(current_node, grid):
            tentative_g_score = current_node.g_score + 1

            if tentative_g_score < neighbor.g_score:
                count += 1
                neighbor.parent = current_node
                neighbor.g_score = tentative_g_score
                neighbor.h_score = heuristic(neighbor, goal)

                heapq.heappush(open_set, (neighbor.g_score + neighbor.h_score, count, neighbor))

    return None

def dijkstra(start, goal, grid):
    open_set = []
    count = 0
    heapq.heappush(open_set, (0, count, start))
    start.g_score = 0

    while open_set:
        current_cost, _, current_node = heapq.heappop(open_set)

        if current_node == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        for neighbor in get_neighbors(current_node, grid):
            tentative_g_score = current_node.g_score + 1

            if tentative_g_score < neighbor.g_score:
                count += 1
                neighbor.parent = current_node
                neighbor.g_score = tentative_g_score

                heapq.heappush(open_set, (neighbor.g_score, count, neighbor))

    return None

def bfs(start, goal, grid):
    queue = deque([start])
    start.g_score = 0

    while queue:
        current_node = queue.popleft()

        if current_node == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        for neighbor in get_neighbors(current_node, grid):
            if neighbor.g_score == float('inf'):
                neighbor.g_score = current_node.g_score + 1
                neighbor.parent = current_node
                queue.append(neighbor)

    return None

def dfs(start, goal, grid):
    stack = [start]
    start.g_score = 0

    while stack:
        current_node = stack.pop()

        if current_node == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        for neighbor in get_neighbors(current_node, grid):
            if neighbor.g_score == float('inf'):
                neighbor.g_score = current_node.g_score + 1
                neighbor.parent = current_node
                stack.append(neighbor)

    return None

class MazeSolverApp:
    def __init__(self, master, width, height):
        self.master = master
        self.master.title("Maze Solver")
        self.solve_thread = None

        # Create main frames for the layout
        self.maze_frame = tk.Frame(self.master, bg='grey')
        self.maze_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.master, bg='lightgrey')
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a canvas in the maze frame
        self.canvas = tk.Canvas(self.maze_frame, width=width, height=height, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.width = width
        self.height = height
        self.cell_size = 20

        self.grid = [[Node(x, y) for y in range(self.height // self.cell_size)] for x in range(self.width // self.cell_size)]

        # Set start and end points
        self.start = self.grid[1][1]
        self.end = self.grid[-2][-2]
        self.start.g_score = 0

        self.create_maze()
        self.draw_grid()

        # Creating buttons
        self.create_buttons()

        self.canvas.bind("<Button-1>", self.draw_obstacle)

    def create_buttons(self):
        # Define button packing options
        button_pack_options = {'side': tk.TOP, 'padx': 5, 'pady': 5, 'expand': True}

        # Create and pack buttons in the button frame
        tk.Button(self.button_frame, text="Solve (A*)", command=self.solve_maze_astar).pack(**button_pack_options)
        tk.Button(self.button_frame, text="Solve (Dijkstra)", command=self.solve_maze_dijkstra).pack(
            **button_pack_options)
        tk.Button(self.button_frame, text="Solve (BFS)", command=self.solve_maze_bfs).pack(**button_pack_options)
        tk.Button(self.button_frame, text="Solve (DFS)", command=self.solve_maze_dfs).pack(**button_pack_options)
        tk.Button(self.button_frame, text="Clear Maze", command=self.clear_maze).pack(**button_pack_options)

        self.canvas.bind("<Button-1>", self.draw_obstacle)

    def draw_grid(self):
        for row in self.grid:
            for node in row:
                color = "black" if node.is_obstacle else "white"
                self.canvas.create_rectangle(
                    node.x * self.cell_size,
                    node.y * self.cell_size,
                    (node.x + 1) * self.cell_size,
                    (node.y + 1) * self.cell_size,
                    fill=color
                )

        self.draw_point(self.start.x, self.start.y, "green")
        self.draw_point(self.end.x, self.end.y, "red")

    def draw_point(self, x, y, color):
        self.canvas.create_oval(
            x * self.cell_size,
            y * self.cell_size,
            (x + 1) * self.cell_size,
            (y + 1) * self.cell_size,
            fill=color
        )

    def create_maze(self):
        for row in self.grid:
            for node in row:
                node.is_obstacle = False
                node.parent = None
                node.g_score = float('inf')
                node.h_score = 0

        obstacle_density = 0.3
        for row in self.grid:
            for node in row:
                if random.random() < obstacle_density and node != self.start and node != self.end:
                    node.is_obstacle = True

        self.start = self.grid[1][1]
        self.end = self.grid[-2][-2]
        self.start.g_score = 0

        self.draw_grid()

    def solve_maze_astar(self):
        self.solve_thread = threading.Thread(target=self._solve_maze, args=(astar, "blue"))
        self.solve_thread.start()

    def solve_maze_dijkstra(self):
        self.solve_thread = threading.Thread(target=self._solve_maze, args=(dijkstra, "green"))
        self.solve_thread.start()

    def solve_maze_bfs(self):
        self.solve_thread = threading.Thread(target=self._solve_maze, args=(bfs, "orange"))
        self.solve_thread.start()

    def solve_maze_dfs(self):
        self.solve_thread = threading.Thread(target=self._solve_maze, args=(dfs, "purple"))
        self.solve_thread.start()

    def _solve_maze(self, algorithm, color):
        for row in self.grid:
            for node in row:
                if not node.is_obstacle:
                    node.parent = None
                    node.g_score = float('inf')
                    node.h_score = 0
        self.start.g_score = 0
        path = algorithm(self.start, self.end, self.grid)
        self.display_path(path, color)

    def display_path(self, path, color):
        if path:
            print(f"Solution Found: {path}")
            for x, y in path:
                self.canvas.after(100, self.draw_point, x, y, color)
        else:
            print("No Solution Found")

    def draw_obstacle(self, event):
        x = event.x // self.cell_size
        y = event.y // self.cell_size

        if 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]):
            self.grid[x][y].is_obstacle = not self.grid[x][y].is_obstacle
            self.draw_grid()

    def clear_maze(self):
        for row in self.grid:
            for node in row:
                node.is_obstacle = False
                node.parent = None
                node.g_score = float('inf')
                node.h_score = 0
        self.start = self.grid[1][1]
        self.end = self.grid[-2][-2]
        self.start.g_score = 0
        self.draw_grid()

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeSolverApp(root, width=500, height=500)
    root.mainloop()
