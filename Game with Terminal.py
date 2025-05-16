import numpy as np
from sklearn.svm import SVC
import random
from typing import List, Tuple, Optional

GRID_SIZE = 10

class Grid:
    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.obstacles: List[Tuple[int, int]] = []

    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size and self.grid[x, y] == 0

    def add_obstacle(self, x: int, y: int) -> bool:
        if self.is_valid(x, y):
            self.grid[x, y] = 1
            self.obstacles.append((x, y))
            return True
        return False

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# --- PSO Algorithm ---
class PSO:
    def __init__(self, start: Tuple[int, int], target: Tuple[int, int], num_particles: int = 30, max_iters: int = 100):
        self.start = start
        self.target = target
        self.num_particles = num_particles
        self.max_iters = max_iters
        self.w, self.c1, self.c2 = 0.5, 1.0, 1.0

        self.particles = np.random.uniform(0, 1, (num_particles, 2))  # speed, angle
        self.velocities = np.random.uniform(-0.1, 0.1, (num_particles, 2))
        self.p_best = self.particles.copy()
        self.p_best_fitness = np.full(num_particles, np.inf)
        self.g_best = self.particles[0].copy()
        self.g_best_fitness = np.inf

    def fitness(self, speed: float, angle: float) -> float:
        dist = manhattan_distance(self.start, self.target)
        time_to_target = dist / max(speed, 1e-5)
        return time_to_target + abs(angle)

    def optimize(self) -> Tuple[float, float]:
        for _ in range(self.max_iters):
            for i in range(self.num_particles):
                speed, angle = self.particles[i]
                fit = self.fitness(speed, angle)

                if fit < self.p_best_fitness[i]:
                    self.p_best_fitness[i] = fit
                    self.p_best[i] = self.particles[i].copy()

                if fit < self.g_best_fitness:
                    self.g_best_fitness = fit
                    self.g_best = self.particles[i].copy()

            r1, r2 = np.random.random((self.num_particles, 2))
            cognitive = self.c1 * r1 * (self.p_best - self.particles)
            social = self.c2 * r2 * (self.g_best - self.particles)
            self.velocities = self.w * self.velocities + cognitive + social
            self.particles += self.velocities
            np.clip(self.particles, 0, 1, out=self.particles)

        return tuple(self.g_best)

# --- ACO Algorithm ---
class ACO:
    def __init__(self, grid: Grid, start: Tuple[int, int], target: Tuple[int, int], num_ants: int = 20, max_iters: int = 100):
        self.grid = grid
        self.start = start
        self.target = target
        self.num_ants = num_ants
        self.max_iters = max_iters
        self.alpha, self.beta = 1.0, 2.0
        self.rho, self.Q = 0.5, 100.0

        # pheromone matrix: 4D for from-to edges
        self.pheromones = np.ones((GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE)) * 0.1

    def neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        possible = [(x+dx, y+dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]]
        return [n for n in possible if self.grid.is_valid(*n)]

    def run(self) -> Optional[List[Tuple[int, int]]]:
        best_path = None
        best_length = np.inf

        for _ in range(self.max_iters):
            all_paths = []
            for _ in range(self.num_ants):
                path = [self.start]
                current = self.start

                while current != self.target:
                    nbrs = self.neighbors(current)
                    if not nbrs:
                        break

                    probs = []
                    x, y = current
                    for nx, ny in nbrs:
                        heuristic = 1.0 / (manhattan_distance((nx, ny), self.target) + 1)
                        pheromone = self.pheromones[x, y, nx, ny]
                        probs.append((pheromone ** self.alpha) * (heuristic ** self.beta))
                    probs = np.array(probs)
                    probs /= probs.sum()

                    next_pos = random.choices(nbrs, probs)[0]
                    path.append(next_pos)
                    current = next_pos

                if current == self.target:
                    all_paths.append(path)

            # Update pheromones
            self.pheromones *= (1 - self.rho)
            for path in all_paths:
                length = len(path) - 1
                if length < best_length:
                    best_length = length
                    best_path = path
                for i in range(len(path) - 1):
                    x1, y1 = path[i]
                    x2, y2 = path[i + 1]
                    self.pheromones[x1, y1, x2, y2] += self.Q / length

        return best_path

# --- SVM Decision ---
class SVMDecision:
    def __init__(self):
        self.model = SVC(kernel='rbf')
        X = [[0.8, 2], [0.2, 2], [0.5, 5], [0.9, 8]]
        y = [1, -1, -1, 1]
        self.model.fit(X, y)

    def decide(self, health: float, distance: float) -> int:
        return self.model.predict([[health, distance]])[0]

# --- Revolutionary Algorithm ---
class RevolutionaryAlgorithm:
    def __init__(self, obstacles: List[Tuple[int, int]], num_generations: int = 100, pop_size: int = 30):
        self.obstacles = obstacles
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.revolution_rate = 0.1

        self.population = np.random.uniform(0, 1, (pop_size, 2))  # move_weight, avoid_weight

    def fitness(self, individual: np.ndarray) -> float:
        move_weight, avoid_weight = individual
        return move_weight * 10 - avoid_weight * len(self.obstacles)

    def run(self) -> np.ndarray:
        for _ in range(self.num_generations):
            fitnesses = np.array([self.fitness(ind) for ind in self.population])

            survivors = self.population[np.argsort(fitnesses)[-self.pop_size // 2:]]

            num_revolutionaries = max(1, int(self.pop_size * self.revolution_rate))
            revolutionaries = np.random.uniform(0, 1, (num_revolutionaries, 2))

            offspring = []
            while len(offspring) < self.pop_size - len(survivors) - num_revolutionaries:
                p1, p2 = random.choices(list(survivors), k=2)
                child = (p1 + p2) / 2 + np.random.normal(0, 0.1, 2)
                offspring.append(np.clip(child, 0, 1))

            self.population = np.vstack([survivors, offspring, revolutionaries])

        fitnesses = np.array([self.fitness(ind) for ind in self.population])
        best_idx = np.argmax(fitnesses)
        return self.population[best_idx]

# --- Perceptron ---
class Perceptron:
    def __init__(self):
        self.weights = np.random.uniform(-1, 1, 2)
        self.bias = 0
        self.learning_rate = 0.1
        self.X = np.array([[0.8, 2], [0.2, 2], [0.5, 5], [0.9, 8]])
        self.y = np.array([1, 0, 0, 1])
        self.train()

    def train(self) -> None:
        for _ in range(100):
            for x_i, target in zip(self.X, self.y):
                pred = 1 if np.dot(self.weights, x_i) + self.bias > 0 else 0
                error = target - pred
                if error != 0:
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error

    def decide(self, health: float, distance: float) -> int:
        return 1 if np.dot(self.weights, [health, distance]) + self.bias > 0 else 0

# --- Main Game Logic ---
def main():
    grid = Grid()

    # User Input
    try:
        target_x = int(input("Enter target x (0-9): "))
        target_y = int(input("Enter target y (0-9): "))
        if not grid.is_valid(target_x, target_y):
            print("Invalid target position!")
            return

        num_obs = int(input("Enter number of obstacles: "))
        print("Enter obstacle positions:")
        for _ in range(num_obs):
            ox, oy = int(input("Obstacle x: ")), int(input("Obstacle y: "))
            if (ox, oy) != (target_x, target_y) and (ox, oy) != (0, 0):
                if not grid.add_obstacle(ox, oy):
                    print(f"Skipping invalid obstacle position ({ox},{oy})")

        health = float(input("Enter agent health (0.0 to 1.0): "))
        max_iters = int(input("Enter iterations for algorithms: "))

    except ValueError:
        print("Invalid input!")
        return

    start = (0, 0)
    target = (target_x, target_y)

    print("\nExecuting AI algorithms...\n")

    # PSO
    pso = PSO(start, target, max_iters=max_iters)
    speed, angle = pso.optimize()
    print(f"PSO: Optimal speed = {speed:.2f}, angle = {angle:.2f}")

    # ACO
    aco = ACO(grid, start, target, max_iters=max_iters)
    path = aco.run()
    if path:
        print(f"ACO: Best path found = {path}")
    else:
        print("ACO: No valid path found")

    # SVM
    svm_decider = SVMDecision()
    distance = manhattan_distance(start, target)
    action = svm_decider.decide(health, distance)
    print(f"SVM: Agent should {'attack' if action == 1 else 'retreat'} (health={health}, distance={distance})")

    # Revolutionary Algorithm
    ra = RevolutionaryAlgorithm(grid.obstacles, num_generations=max_iters)
    move_w, avoid_w = ra.run()
    print(f"Revolutionary Algorithm: Strategy weights -> move: {move_w:.2f}, avoid: {avoid_w:.2f}")

    # Perceptron
    perceptron = Perceptron()
    pursue = perceptron.decide(health, distance)
    print(f"Perceptron: Agent should {'pursue' if pursue == 1 else 'not pursue'} the target")

if __name__ == "__main__":
    main()

