import numpy as np
from matplotlib import pyplot as plt
from numpy.core._multiarray_umath import ndarray


class Solver_TSP:

    solution: ndarray
    found_length: float

    def __init__(self, method):
        self.available_methods = {"random": self.random_method, "nearest_neighbors": self.nn}
        self.method = method
        self.solved = False
        assert method in self.available_methods, f"the {method} method is not available currently."

    def __call__(self, instance_, verbose=True, return_value=True):
        self.instance = instance_
        if verbose:
            print(f"###  solving with {self.method}####")
        self.solution = self.available_methods[self.method](instance_)
        assert self.check_if_solution_is_valid(self.solution), "Error the solution is not valid"
        self.evaluate_solution()
        self._gap()
        if verbose:
            print(f"###  solution found with {self.gap} ####")
        self._gap()
        if return_value:
            return self.solution

    def random_method(self, instance_):
        n = int(instance_.nPoints)
        solution = np.random.choice(np.arange(n), size=n, replace=False)
        self.solution = np.concatenate([solution, [solution[0]]])
        self.solved = True
        return self.solution

    def nn(self, instance_, starting_node=0):
        dist_matrix = np.copy(instance_.dist_matrix)
        n = int(instance_.nPoints)
        node = np.argmin([starting_node])
        tour = [node]
        for _ in range(n - 2):
            for node in np.argsort(dist_matrix[node]):
                if node not in tour:
                    tour.append(node)
        tour.append(starting_node)
        self.solution = np.array(tour)
        self.solved = True
        return self.solution

    def plot_solution(self):
        assert self.solved, "You can't plot the solution, you need to solve it first!"
        plt.figure(figsize=(8, 8))
        plt.title(f"{self.instance.name} with gap {self.gap}")
        ordered_points = self.instance.points[self.solution]
        plt.plot(ordered_points[:, 1], ordered_points[:, 2], 'b-')
        plt.show()

    def check_if_solution_is_valid(self, solution):
        rights_values = np.sum([self.check_validation(i, solution[:-1]) for i in np.arange(self.instance.nPoints)])
        if rights_values == self.instance.nPoints:
            return True
        else:
            return False

    def check_validation(self, node, solution):
        if np.sum(solution == node) == 1:
            return 1
        else:
            return 0

    def evaluate_solution(self, return_value=False):
        total_length = 0
        starting_node = self.solution[0]
        from_node = starting_node
        for node in self.solution[1:]:
            total_length += self.instance.dist_matrix[from_node, node]
            from_node = node

        self.found_length = total_length
        if return_value:
            return total_length

    def _gap(self):
        self.evaluate_solution(return_value=False)
        self.gap = np.round((self.found_length - self.instance.best_sol) / self.instance.best_sol * 100, 2)