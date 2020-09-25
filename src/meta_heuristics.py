import numpy as np
import os
if 'AI' in os.getcwd():
    from src.utils import *
else:
    from AI2019.src.utils import *


class Simulated_Annealing:

    @staticmethod
    def sa(solution, instance, constant_temperature=0.95, iterations_for_each_temp=100):

        # initial setup
        temperature = instance.best_sol / np.sqrt(instance.nPoints)
        current_sol = np.array(solution)
        current_len = compute_length(solution, instance.dist_matrix)
        best_sol = np.array(solution)
        best_len = current_len

        # main loop
        while temperature > 0.001:
            for it in range(iterations_for_each_temp):
                next_sol, delta_E = Simulated_Annealing.random_sol_from_neig(current_sol, instance)
                if delta_E < 0:
                    current_sol = next_sol
                    current_len += delta_E
                    if current_len < best_len:
                        best_sol = current_sol
                        best_len = current_len
                else:
                    r = np.random.uniform(0, 1)
                    if r < np.exp(- delta_E / temperature):
                        current_sol = next_sol
                        current_len += delta_E

            temperature *= constant_temperature

        return best_sol.tolist()

    @staticmethod
    def random_sol_from_neig(solution, instance):
        i, j = np.random.choice(np.arange(1, len(solution) - 1), 2, replace=False)
        i, j = np.sort([i, j])
        return Simulated_Annealing.swap2opt(solution, i, j), Simulated_Annealing.gain(i, j, solution,
                                                                                      instance.dist_matrix)

    @staticmethod
    def swap2opt(tsp_sequence, i, j):
        new_tsp_sequence = np.copy(tsp_sequence)
        new_tsp_sequence[i:j + 1] = np.flip(tsp_sequence[i:j + 1], axis=0)  # flip or swap ?
        return new_tsp_sequence

    @staticmethod
    def gain(i, j, tsp_sequence, matrix_dist):
        old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[i - 1]] + matrix_dist[
            tsp_sequence[j], tsp_sequence[j + 1]])
        changed_links_len = (matrix_dist[tsp_sequence[j], tsp_sequence[i - 1]] + matrix_dist[
            tsp_sequence[i], tsp_sequence[j + 1]])
        return - old_link_len + changed_links_len


class Iterated_Local_Search:

    def __call__(self):
        pass
