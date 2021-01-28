"""
  Created by mohammed_elkomy

"""
import heapq
import os
import sys
from multiprocessing import Pool
from random import randrange

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand, normal
from opteval import benchmark_func as bf

PATH_IMGS_3D = "3d_imgs"
import copy


def plt_to_img(fig):
    """
    converts matplotlib fig into array
    :param fig: matplotlib fig
    :return: numpy array RGB
    """
    fig.canvas.draw()
    draw = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    draw = draw.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    return draw


class Individual:
    def __init__(self, upper, lower, dim, fitness_callback, state=None):
        self.__upper = upper
        self.__lower = lower
        self.__state_vector = state
        if self.__state_vector is None:
            self.__state_vector = lower + rand(dim) * (upper - lower)
        else:
            self.__state_vector = self.boundary_check()

        self.__fitness = fitness_callback(self.__state_vector)
        self.fitness_callback = fitness_callback

    def update(self):
        self.__state_vector = self.boundary_check()
        self.__fitness = self.fitness_callback(self.__state_vector)

    def boundary_check(self):
        # clipping out of bound values
        return np.minimum(np.maximum(self.__lower, self.__state_vector), self.__upper)

    def dim(self):
        return self.__state_vector.shape[0]

    @property
    def state(self):
        return self.__state_vector

    @property
    def fitness(self):
        return self.__fitness

    def __hash__(self):
        return hash(self.__fitness)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__fitness == other.fitness


class StochasticFractalSearch:
    """
    *************************************************************************
     Stochastic Fractal Search Algorithm
    *************************************************************************
    """

    def __init__(self, lower, upper, maximum_diffusion, population_size, walk, fitness_callback, max_generation, bound_clipping=True, demo_3d=False, plot_step=.5, refresh_inter=.01, iter_callback=None, perturb=.2):
        """
        a class for SFS algorithm
        :param bound_clipping: true to clip out of bound values | false to use random values as in the original algorithm on matlab
        :param iter_callback: the function to be called in every optimization iteration (should receive generation number,)
        :param lower: a numpy vector for the lower bound of size (dim,), the problem dimension is set from this
        :param upper: a numpy vector for the upper bound of size (dim,), can't be less then lower in any entry
        :param maximum_diffusion: Maximum Diffusion Number (MDN) considered as Maximum_Diffusion 0 ~ 2 (for the diffusion process)
        :param population_size: the size of the solution population
        :param walk: Choosing diffusion walk considered as Walk
            Walk = 1 ----> SFS uses the first Gaussian walk(usually SIMPLE Problems)
            Walk = 0 ----> SFS uses the second Gaussian walk(usually HARD Problems)

            You can also write:
            Walk = 0.75 ---> SFS uses the first Gaussian walk, with probability
            of 75% which comes from uniform, and SFS uses the second Gaussian walk
            distribution with probability of 25% .
            Generally, to solve your problem, try to use both walks.
            If you want to use the second Gaussian walk please increase the maximum generation

        :param fitness_callback: the function to be called for each point to return the fitness
        :param max_generation: considered as Maximum_Generation
        :param demo_3d: boolean to enable 3d demo for 2d problems
        :param plot_step: the step for the meshgrid of the 3d drawing
        :param refresh_inter: interval by which the drawer pauses in each iteration
        """
        ##############################################################
        # initialization steps
        self.iter_callback = iter_callback

        if np.any(lower > upper):
            raise ValueError("please check your bounds, lower is greater the upper")
        """
        self.Lower: the lower limit vector (a vector of the size : state space dimension (Dim) ) (float32)
        self.upper: the upper limit vector (a vector of the size : state space dimension (Dim) ) (float32)
        """
        self.lower = lower
        self.upper = upper

        self.dim = lower.shape[0]  # problem dimension
        self.maximum_diffusion = maximum_diffusion
        # Creating random points in considered search space
        self.population_size = population_size
        self.fitness_callback = fitness_callback
        self.generation = 1
        self.max_generation = max_generation
        self.walk = walk
        self.bound_clipping = bound_clipping
        self.population = [self.new_individual() for _ in range(population_size)]
        self.perturb = perturb
        ##############################################################
        # initializing the best point
        # sort the population based on the fitness
        self.population = self.sort_individuals(self.population)
        # Finding the Best point in the group
        self.best_point = self.population[0]
        ##############################################################
        # drawing
        self.draw_3d = demo_3d
        self.last_draw = np.inf
        self.plot_step = plot_step
        self.pause_mode = False
        # 3d
        if demo_3d:
            self.fig_3d = plt.figure()
            self.demo_3d_ax = Axes3D(self.fig_3d)
            self.fig_3d.canvas.mpl_connect('key_press_event', self.press)
            self.collection_3d = None
            self.last_XYZ = None  # for smooth animation
            self.anim_index = 0
            self.anim_resolution = (self.upper - self.lower) / 1000
            self.pause_mode = True

        # fitness
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        self.fitness_ax = ax
        self.fitness_fig = fig
        self.fitness_fig.canvas.mpl_connect('key_press_event', self.press)
        self.refresh_inter = refresh_inter

        ##############################################################

    def new_individual(self, state=None):
        return Individual(self.upper, self.lower, self.dim, self.fitness_callback, state)

    def press(self, event):
        """
        callback for key press to enable or disable pause mode
        :param event: matplotlib event
        """
        sys.stdout.flush()
        if event.key == 'd':
            print('Pause Mode', self.pause_mode)
            self.pause_mode = not self.pause_mode

    @staticmethod
    def sort_individuals(individuals):
        """
        sorting individuals list on fitness
        :return: sorted individuals
        """
        return sorted(set(individuals), key=lambda item: item.fitness)

    def update_best(self, population_sorted):
        """
        update the best tracked fitness
        :param population_sorted: sorted population array 2d
        """
        if population_sorted[0].fitness < self.best_point.fitness:
            self.best_point = population_sorted[0]

    def diffusion_process(self, source):
        """
        This function is used to mimic diffusion process, and creates some new points based on Gaussian Walks.
        :param source: the input point which is going to be diffused
        :return: a tuple (the new point created by Diffusion process , the value of fitness function)
        """
        # calculating the maximum diffusion for each point
        num_diffusion = self.maximum_diffusion
        dimension = source.dim()

        new_points = [source]
        for i in range(num_diffusion):
            # consider which walks should be selected
            sigma = (np.log(self.generation + 1) / self.generation) * (abs((source.state - self.best_point.state)))  # eqn (13)
            if rand() < self.walk:
                # simple walk, gaussian around best
                generated_point = normal(loc=self.best_point.state, scale=sigma, size=(dimension,)) + (rand() * self.best_point.state - rand() * source.state)  # eqn (11),a mistake in the code replacing uniform and normal distributions
            else:
                # complex walk, gaussian around Point
                generated_point = normal(loc=source.state, scale=sigma, size=(dimension,))  # eqn (12)
            new_points.append(self.new_individual(generated_point))

        return new_points

    def update_2_step(self, pa_sorted):
        """
        the update 2 step as described in the paper
        :param pa_sorted: probabilities sorted for each point (assuming population is sorted)
        :return: a tuple( updated population based on the randomized step, update 2 fitness as well)
        """
        population_update2 = []
        perturb_idx = int(self.perturb * len(self.population))
        for point_idx in range(perturb_idx):
            if rand() > pa_sorted[point_idx]:
                r1 = randrange(self.population_size)
                r2 = randrange(self.population_size)
                while r2 == r1:
                    r2 = randrange(self.population_size)

                if rand() < .5:
                    replace_point = self.population[point_idx].state - rand() * (self.population[r2].state - self.best_point.state)  # eqn (17)
                else:
                    replace_point = self.population[point_idx].state - rand() * (self.population[r2].state - self.population[r1].state)  # eqn (18)
                replace_point = self.new_individual(replace_point)

                population_update2.append(replace_point)

        return population_update2

    def update_1_step(self, pa_sorted):
        """
        :param pa_sorted: probabilities sorted for each point (assuming population is sorted)
        :return: a tuple(updated population based on the randomized step, update 2 fitness )
        """
        rand_vec1 = np.random.permutation(self.population_size)
        rand_vec2 = np.random.permutation(self.population_size)

        population = copy.deepcopy(self.population)
        population_update1 = []
        # population = self.population.copy()
        perturb_idx = int(self.perturb * len(self.population))
        for point_idx in range(perturb_idx):
            change = False
            for component_idx in range(self.dim):
                if rand() > pa_sorted[point_idx]:
                    change = True
                    source_point = self.population[point_idx].state
                    target_point = population[point_idx].state
                    rand_point1 = self.population[rand_vec1[point_idx]].state
                    rand_point2 = self.population[rand_vec2[point_idx]].state
                    target_point[component_idx] = rand_point1[component_idx] - rand() * (rand_point2[component_idx] - source_point[component_idx])  # eqn(16)
                # else :# leave unchanged
            if change:
                population[point_idx].update()
                population_update1.append(population[point_idx])
        return population_update1

    def diffusion_step(self):
        """
        for each point in the population apply the diffusion as described in the paper
        :return:  a tuple(diffused population based on the diffusion_process step, fitness for the new diffused population)
        """
        perturb_idx = int(self.perturb * len(self.population))
        population_diffusion = []
        for candidate in self.population[:perturb_idx]:
            # creating new points based on diffusion process >> exploitation
            new_points = self.diffusion_process(candidate)
            population_diffusion.extend(new_points)
        return population_diffusion

    def optimize(self):
        """
        the optimization process, for number of max_generation
        """
        # starting the Optimizer
        for _ in range(self.max_generation):
            self.iterate()
        # clean up
        plt.close('all')

    def iterate(self):
        ################################################################################################
        # 1.a) diffusion process  >> exploitation
        ################################################################################################
        population_diffusion = self.diffusion_step()
        population_diffusion_sorted = self.sort_individuals(population_diffusion)
        # clean up and make sure they aren't used again (might be a source for confusion)
        del population_diffusion
        #self.population = population_diffusion_sorted[:self.population_size]  # clip
        ################################################################################################
        # 1.b) update the best point and fitness
        ################################################################################################
        self.update_best(self.population)
        ################################################################################################
        # 1.c) ranking after diffusion process
        ################################################################################################
        # for simplicity the population will be sorted, so the probabilities are a linear function with sorted output (line:54 in matlab script)
        pa_sorted = [(self.population_size - i) / self.population_size for i in range(self.population_size)]  # probabilities for points, ranking eqn(15)
        ################################################################################################
        ################################################################################################
        ################################################################################################
        ################################################################################################
        # 2) update 1 : updating the best points obtained by diffusion process  >> exploration
        ################################################################################################

        population_update1 = self.update_1_step(pa_sorted)  # population is sorted
        # keep the best of both the updated and diffused (merging points form 2 populations)
        self.population.extend(population_update1)
        self.population = self.sort_individuals(self.population)

        # self.population = population_diffusion_sorted[:self.population_size] # clip
        # if len(set(self.population)) != self.population_size:
        #     print()
        ################################################################################################
        # 2.b)  update the best point and fitness
        ################################################################################################
        self.update_best(self.population)
        ################################################################################################
        ## 2.c) ranking after update1 process
        ################################################################################################
        # probabilities for points, ranking eqn(15)
        # pa_sorted = same as before update 1
        ################################################################################################
        ################################################################################################
        ################################################################################################
        ################################################################################################
        # 3) update 2 : updating the best points obtained by diffusion process  >> exploration
        ################################################################################################
        population_update2 = self.update_2_step(pa_sorted)
        self.population.extend(population_update2)
        self.population = self.sort_individuals(self.population)
        self.population = self.population[:self.population_size]
        # if len(set(self.population)) != self.population_size:
        #     print()
        ################################################################################################
        ################################################################################################
        ################################################################################################
        ################################################################################################
        # 4) drawing stuff
        self.fitness_ax.plot([self.generation - 1, self.generation], [self.last_draw, self.best_point.fitness])

        self.last_draw = self.best_point.fitness  # used to connect lines
        plt_to_img(self.fitness_fig)
        if self.draw_3d:
            # for 3d drawing for dim = 2 problems
            draw_3d_imgs = self.demo_3d(self.population)
            for img in draw_3d_imgs:
                cv2.imwrite(os.path.join(PATH_IMGS_3D, "anim-{}.png".format(self.anim_index)), img)
                self.anim_index += 1
        self.generation += 1
        plt.pause(self.refresh_inter)
        if self.iter_callback is not None:
            self.iter_callback(self.best_point, self.generation)

    def demo_3d(self, population):
        """
        draws a 3d image for the current generation
        :param population: population array 2d
        :param fitness: the fitness for each point in the population (a vector)
        :return: plt_to_img results
        """

        X_p, Y_p, Z_p, plt_to_img_results = [], [], [], []
        for individual in population:
            X_p.append(individual.state[0])
            Y_p.append(individual.state[1])
            Z_p.append(individual.fitness)

        X_p, Y_p, Z_p = np.array(X_p), np.array(Y_p), np.array(Z_p)

        if self.collection_3d is None:
            # draw using meshgrid
            x = np.arange(self.lower[0], self.upper[0], self.plot_step, dtype=np.float32)
            y = np.arange(self.lower[1], self.upper[1], self.plot_step, dtype=np.float32)
            X, Y = np.meshgrid(x, y)
            Z = []
            for xy_list in zip(X, Y):
                z = []
                for xy_input in zip(xy_list[0], xy_list[1]):
                    tmp = list(xy_input)
                    # tmp.extend(list(self.optimal_solution[0:self.dim - 2]))
                    z.append(self.fitness_callback(np.array(tmp)))
                Z.append(z)
            Z = np.array(Z)
            self.demo_3d_ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=.7,
                                         linewidth=.1, antialiased=True)

            # self.demo_ax.plot_wireframe(X, Y, Z)
            self.collection_3d = self.demo_3d_ax.scatter(X_p, Y_p, Z_p, marker='o', linewidth=5, color='red')
        else:
            # draw points with animation
            X_last, Y_last, Z_last = self.last_XYZ
            X_diff, Y_diff, Z_diff = X_p - X_last, Y_p - Y_last, Z_p - Z_last

            resol_x, resol_y = self.anim_resolution
            steps = max(-int(np.mean(X_diff / resol_x + Y_diff / resol_y)), 3)
            X_diff, Y_diff, Z_diff = X_diff / steps, Y_diff / steps, Z_diff / steps
            for i in range(steps):  # animation steps
                self.collection_3d._offsets3d = X_last + i * X_diff, Y_last + i * Y_diff, Z_last + i * Z_diff
                plt_to_img_results.append(plt_to_img(self.fig_3d))

                # pause mode
                while self.pause_mode:
                    plt.pause(self.refresh_inter)
                plt.pause(self.refresh_inter)
            self.collection_3d._offsets3d = X_p, Y_p, Z_p
            plt.pause(.1)

        self.last_XYZ = (X_p, Y_p, Z_p)
        return plt_to_img_results


if __name__ == "__main__":
    print(bf.__all__)
    #########################################################################
    problem = bf.Ackley(2)
    problem = bf.StyblinskiTang(2)
    # problem = bf.BukinN6()
    lower = problem.min_search_range
    upper = problem.max_search_range

    print(problem.get_global_optimum_solution(), problem.optimal_solution)
    sfs = StochasticFractalSearch(lower, upper, 2, 50, .5, problem.get_func_val, 500, demo_3d=True)
    sfs.optimize()
    #########################################################################
    # problem = bf.Easom()
    # lower = problem.min_search_range
    # upper = problem.max_search_range
    #
    # print(problem.get_global_optimum_solution(), problem.optimal_solution)
    # sfs = StochasticFractalSearch(lower, upper, 2, 50, .5, problem.get_func_val, 30, demo_3d=True, refresh_inter=.5, plot_step=.5)
    # sfs.optimize()
    #########################################################################
    # problem = bf.StyblinskiTang(10)
    # lower = problem.min_search_range
    # upper = problem.max_search_range
    # print("actual optimal value", problem.get_global_optimum_solution())
    # print("actual optimal point", problem.optimal_solution)
    # sfs = StochasticFractalSearch(lower, upper, 10, 50, .5, problem.get_func_val, 100, demo_3d=False)
    # sfs.optimize()
    # print("algorithm optimal value", sfs.best_point.fitness)
    # print("algorithm optimal point", sfs.best_point)
