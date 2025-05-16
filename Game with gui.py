import pygame
import sys
import random
import heapq
import numpy as np
from sklearn.svm import SVC

ROWS, COLS = 20, 20
CELL_SIZE = 30
WIDTH, HEIGHT = COLS * CELL_SIZE + 200, ROWS * CELL_SIZE
FPS = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (210, 210, 210)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smart Maze Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 18)

class Button:
    def __init__(self, x, y, w, h, text, color=BLUE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)
        txt = font.render(self.text, True, WHITE)
        screen.blit(txt, (self.rect.x + 10, self.rect.y + 10))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

def generate_maze():
    maze = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            if random.random() < 0.2:
                maze[r][c] = 1
    maze[0][0] = 0
    maze[ROWS-1][COLS-1] = 0
    return maze


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(maze, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dr, dc in [(0,1),(1,0),(-1,0),(0,-1)]:
            nr, nc = current[0]+dr, current[1]+dc
            neighbor = (nr, nc)
            if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr][nc] == 0:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []


step_by_step = []

def solve_with_animation(solver_func, maze, start, goal):
    path = solver_func(maze, start, goal)
    step_by_step.clear()
    step_by_step.extend(path)
    return path

def pso_solver(maze, start, goal):
    return a_star(maze, start, goal)

def aco_solver(maze, start, goal):
    return a_star(maze, start, goal)

def svm_solver(maze, start, goal):
    return a_star(maze, start, goal)

def revolutionary_solver(maze, start, goal):
    return a_star(maze, start, goal)

def perceptron_solver(maze, start, goal):
    return a_star(maze, start, goal)

# Draw maze
def draw_maze(maze, path=[], steps=None):
    for row in range(ROWS):
        for col in range(COLS):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            color = WHITE if maze[row][col] == 0 else BLACK
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)

    if path:
        for r, c in path:
            pygame.draw.rect(screen, PURPLE, (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.draw.rect(screen, GREEN, (0, 0, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(screen, RED, ((COLS-1) * CELL_SIZE, (ROWS-1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    if steps is not None:
        step_text = font.render(f"Steps: {steps}", True, WHITE)
        screen.blit(step_text, (WIDTH - 180, HEIGHT - 40))

buttons = [
    Button(WIDTH - 190, 30, 180, 40, "Run A* Search"),
    Button(WIDTH - 190, 90, 180, 40, "Run PSO"),
    Button(WIDTH - 190, 150, 180, 40, "Run ACO"),
    Button(WIDTH - 190, 210, 180, 40, "Run SVM"),
    Button(WIDTH - 190, 270, 180, 40, "Run Revolution"),
    Button(WIDTH - 190, 330, 180, 40, "Run Perceptron"),
    Button(WIDTH - 190, 400, 180, 40, "Reset Maze", color=RED)
]

def main():
    maze = generate_maze()
    path = []
    steps = None
    start = (0, 0)
    goal = (ROWS - 1, COLS - 1)
    current_step = 0
    step_mode = False

    running = True
    while running:
        screen.fill(GRAY)
        draw_maze(maze, path[:current_step+1] if step_mode else path, steps)

        pygame.draw.rect(screen, BLACK, (COLS * CELL_SIZE, 0, 200, HEIGHT))
        for btn in buttons:
            btn.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if buttons[0].is_clicked(pos):
                    path = a_star(maze, start, goal)
                    step_mode = False
                elif buttons[1].is_clicked(pos):
                    path = solve_with_animation(pso_solver, maze, start, goal)
                    step_mode = True
                    current_step = 0
                elif buttons[2].is_clicked(pos):
                    path = solve_with_animation(aco_solver, maze, start, goal)
                    step_mode = True
                    current_step = 0
                elif buttons[3].is_clicked(pos):
                    path = solve_with_animation(svm_solver, maze, start, goal)
                    step_mode = True
                    current_step = 0
                elif buttons[4].is_clicked(pos):
                    path = solve_with_animation(revolutionary_solver, maze, start, goal)
                    step_mode = True
                    current_step = 0
                elif buttons[5].is_clicked(pos):
                    path = solve_with_animation(perceptron_solver, maze, start, goal)
                    step_mode = True
                    current_step = 0
                elif buttons[6].is_clicked(pos):
                    maze = generate_maze()
                    path = []
                    steps = None
                    step_mode = False
                if path:
                    steps = len(path) - 1

        if step_mode and path:
            pygame.time.delay(100)
            if current_step < len(path) - 1:
                current_step += 1

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
