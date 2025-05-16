# Smart Maze Game – AI-Based Pathfinding Project
The Smart Maze Game is a Python application demonstrating AI pathfinding and decision-making through two versions:

- **GUI Version:** Uses Pygame to generate and solve random mazes with multiple AI algorithms.
- **Terminal Version:** Simulates an agent navigating a grid with obstacles using various AI methods.

The project showcases integration of search algorithms, optimization techniques, and machine learning models for intelligent pathfinding and strategy.

---

## Features

### GUI Version
- Random 20x20 maze generation with walls and open paths.
- Start at top-left cell, goal at bottom-right.
- Interactive buttons to run different AI solvers:
  - A* Search (implemented)
  - PSO, ACO, SVM, Revolutionary Algorithm, Perceptron
- Visual display of maze, path, and steps.
- Step-by-step path animation.

### Terminal Version
- 10x10 grid with user-defined obstacles and target.
- AI algorithms implemented:
  - Particle Swarm Optimization (PSO) for movement optimization.
  - Ant Colony Optimization (ACO) for pathfinding.
  - Support Vector Machine (SVM) for attack/retreat decision.
  - Revolutionary Algorithm for evolving movement strategy weights.
  - Perceptron for pursue/not pursue decision.
- Console-based inputs and output of algorithm results.

---

## Technologies Used

- Python 3.x
- Pygame (GUI and graphics)
- NumPy (array operations)
- Scikit-learn (SVM model)
- Heapq (priority queue for A*)

---

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed.  
Install dependencies via pip:

```bash
pip install pygame numpy scikit-learn

