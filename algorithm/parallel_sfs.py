"""
  Created by mohammed_elkomy

"""
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


def point_diffusion(args):
    # consider which walks should be selected
    source, best_point, generation, dimension, walk = args
    sigma = (np.log(generation) / generation) * (abs((source - best_point)))  # eqn (13)
    if rand() < walk:
        # simple walk, gaussian around best
        generated_point = normal(loc=best_point, scale=sigma, size=(1, dimension)) + (rand() * best_point - rand() * source)  # eqn (11),a mistake in the code replacing uniform and normal distributions
    else:
        # complex walk, gaussian around Point
        generated_point = normal(loc=source, scale=sigma, size=(1, dimension))  # eqn (12)

    return generated_point


class StochasticFractalSearch():
    """
    *************************************************************************
     Stochastic Fractal Search Algorithm
    *************************************************************************
    """

    def __init__(self, lower, upper, maximum_diffusion, population_size, walk, fitness_callback, max_generation, demo_3d=False, plot_step=.5, refresh_inter=.01, iter_callback=None):
        """
        a class for SFS algorithm
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
        self.population = np.tile(lower, reps=(population_size, 1)) + rand(population_size, self.dim) * np.tile(upper, reps=(population_size, 1))
        self.population_size = population_size
        self.fitness_callback = fitness_callback
        self.generation = 1
        self.max_generation = max_generation
        self.walk = walk
        ##############################################################
        # initializing the best point
        # Calculating the fitness of first created points
        initial_fitness = self.batch_fitness(self.population)
        # sorting on fitness
        sort_index = np.argsort(initial_fitness)
        # sort the population based on the fitness
        self.population = self.population[sort_index]
        # Finding the Best point in the group
        self.best_point = self.population[0]
        self.best_fit = initial_fitness[0]
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
    def simultaneous_fitness_sort(population, fitness):
        """
        sorting both of them based on fitness
        :param population: the population array 2d
        :param fitness: a vector for fitness value for each points
        :return: simultaneously sorted population + fitness
        """
        new_fitness_index = np.argsort(fitness)
        fitness_sorted = fitness[new_fitness_index]
        population_sorted = population[new_fitness_index]
        return population_sorted, fitness_sorted

    def update_best(self, population_sorted, fitness_sorted):
        """
        update the best tracked fitness
        :param population_sorted: sorted, population array 2d
        :param fitness_sorted: a sorted vector for fitness value for each points
        """
        if fitness_sorted[0] < self.best_fit:
            self.best_point = population_sorted[0]
            self.best_fit = fitness_sorted[0]

    def batch_fitness(self, points):
        """
        :param points: the points for which fitness should be computed
        :return: the fitness for the points
        """
        with Pool(processes=16) as pool:
            result = np.array(pool.map(self.fitness_callback, points))

        return result

    def bound_checking(self, points):
        """
        This function is used for SFS problem bound checking
        check the point is in the upper and lower limits and updates the points to be within the limits
        :param points: the points to be checked (2d narray : N X Dim) (float32) OR a single point

        self.Lower: the lower limit vector (a vector of the size : state space dimension (Dim) ) (float32)
        self.upper: the upper limit vector (a vector of the size : state space dimension (Dim) ) (float32)
        """
        single_point = len(points.shape) != 2
        if single_point:
            points = np.expand_dims(points, 0)

        # # RANDOM OUT OF BOUND

        # clipped OUT OF BOUND
        bad_point_points = np.argwhere(np.logical_or(points < self.lower, points > self.upper))  # below_lower+above_upper

        candidate = bad_point_points[:, 0]
        component = bad_point_points[:, 1]
        points[candidate, component] = np.minimum(np.maximum(points[candidate, component], self.lower[component]), self.upper[component])

        if single_point:
            points = points[0, :]
        return points

    def diffusion_process(self, source):
        """
        This function is used to mimic diffusion process, and creates some new points based on Gaussian Walks.
        :param source: the input point which is going to be diffused
        :return: a tuple (the new point created by Diffusion process , the value of fitness function)
        """
        # calculating the maximum diffusion for each point
        num_diffusion = self.maximum_diffusion
        dimension = source.shape[0]

        params = []
        for _ in range(num_diffusion):
            params.append((source, self.best_point, self.generation, dimension, self.walk))

        with Pool(processes=16) as pool:
            new_points = [source.reshape((1, -1))] + pool.map(point_diffusion, params)

        new_points = np.concatenate(new_points, axis=0)
        new_points = self.bound_checking(new_points)  # eqn (14)
        new_points_fitness = self.batch_fitness(new_points)
        new_points_sorted, new_points_fitness_sorted = self.simultaneous_fitness_sort(new_points, new_points_fitness)

        # the newly created pointed is the best one (minimization = minimum fitness)
        create_point_fitness = new_points_fitness_sorted[0]
        create_point = new_points_sorted[0]
        return create_point, create_point_fitness

    def update_2_step(self, pa_sorted):
        """
        the update 2 step as described in the paper
        :param pa_sorted: probabilities sorted for each point (assuming population is sorted)
        :return: a tuple( updated population based on the randomized step, update 2 fitness as well)
        """
        for point_idx in range(self.population_size):
            if rand() > pa_sorted[point_idx]:
                r1 = randrange(self.population_size)
                r2 = randrange(self.population_size)
                while r2 == r1:
                    r2 = randrange(self.population_size)

                if rand() < .5:
                    replace_point = self.population[point_idx] - rand() * (self.population[r2] - self.best_point)  # eqn (17)
                else:
                    replace_point = self.population[point_idx] - rand() * (self.population[r2] - self.population[r1])  # eqn (18)

                replace_point = self.bound_checking(replace_point)

                if self.fitness_callback(replace_point) < self.fitness_callback(self.population[point_idx]):
                    self.population[point_idx] = replace_point
        fitness_update2 = self.batch_fitness(self.population)
        return self.population, fitness_update2

    def update_1_step(self, pa_sorted):
        """
        :param pa_sorted: probabilities sorted for each point (assuming population is sorted)
        :return: a tuple(updated population based on the randomized step, update 2 fitness )
        """
        rand_vec1 = np.random.permutation(self.population_size)
        rand_vec2 = np.random.permutation(self.population_size)
        population_update1 = self.population.copy()
        for point_idx in range(self.population_size):
            for component_idx in range(self.dim):
                if rand() > pa_sorted[point_idx]:
                    population_update1[point_idx, component_idx] = self.population[rand_vec1[point_idx], component_idx] \
                                                                   - rand() * (self.population[rand_vec2[point_idx], component_idx] - self.population[point_idx, component_idx])  # eqn(16)
                # else :# leave unchanged
        population_update1 = self.bound_checking(population_update1)
        fitness_update1 = self.batch_fitness(population_update1)
        return population_update1, fitness_update1

    def diffusion_step(self):
        """
        for each point in the population apply the diffusion as described in the paper
        :return:  a tuple(diffused population based on the diffusion_process step, fitness for the new diffused population)
        """
        population_diffusion, fitness_diffusion = [], []
        for candidate in self.population:
            # creating new points based on diffusion process >> exploitation
            new_point, new_point_fitness = self.diffusion_process(candidate)
            population_diffusion.append(new_point)
            fitness_diffusion.append(new_point_fitness)
        population_diffusion = np.array(population_diffusion)
        fitness_diffusion = np.array(fitness_diffusion)
        return population_diffusion, fitness_diffusion

    def optimize(self):
        """
        the optimization process, for number of max_generation
        """
        # starting the Optimizer
        for _ in range(self.max_generation):
            ################################################################################################
            # 1.a) diffusion process  >> exploitation
            ################################################################################################
            population_diffusion, fitness_diffusion = self.diffusion_step()
            population_diffusion_sorted, fitness_diffusion_sorted = self.simultaneous_fitness_sort(population_diffusion, fitness_diffusion)
            # clean up and make sure they aren't used again (might be a source for confusion)
            del population_diffusion, fitness_diffusion
            ################################################################################################
            # 1.b) update the best point and fitness
            ################################################################################################
            self.update_best(population_diffusion_sorted, fitness_diffusion_sorted)
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
            self.population = population_diffusion_sorted
            population_update1, fitness_update1 = self.update_1_step(pa_sorted)  # population is sorted

            # keep the best of both the updated and diffused (merging points form 2 populations)
            fitness_merged = []
            for idx, (diff_point, update1_point) in enumerate(zip(population_diffusion_sorted, population_update1)):
                if fitness_update1[idx] < fitness_diffusion_sorted[idx]:  # update_1 point is better than diffusion
                    self.population[idx] = update1_point
                    fitness_merged.append(fitness_update1[idx])
                else:
                    self.population[idx] = diff_point  # diffusion point is better than update_1
                    fitness_merged.append(fitness_diffusion_sorted[idx])
            fitness_merged = np.array(fitness_merged)
            self.population, fitness_merged = self.simultaneous_fitness_sort(self.population, fitness_merged)
            ################################################################################################
            # 2.b)  update the best point and fitness
            ################################################################################################
            self.update_best(self.population, fitness_merged)
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
            self.population, fitness_update2 = self.update_2_step(pa_sorted)
            self.population, fitness_update2 = self.simultaneous_fitness_sort(self.population, fitness_update2)

            self.fitness_ax.plot([self.generation - 1, self.generation], [self.last_draw, self.best_fit])

            ################################################################################################
            ################################################################################################
            ################################################################################################
            ################################################################################################
            # 4) drawing stuff
            self.last_draw = self.best_fit  # used to connect lines
            plt_to_img(self.fitness_fig)

            if self.draw_3d:
                # for 3d drawing for dim = 2 problems
                draw_3d_imgs = self.demo_3d(self.population, fitness_update2)
                for img in draw_3d_imgs:
                    cv2.imwrite(os.path.join(PATH_IMGS_3D, "anim-{}.png".format(self.anim_index)), img)
                    self.anim_index += 1

            self.generation += 1
            plt.pause(self.refresh_inter)
            if self.iter_callback is not None:
                self.iter_callback(self.best_point, self.best_fit, self.generation)
        # clean up
        plt.close('all')

    def demo_3d(self, population, fitness):
        """
        draws a 3d image for the current generation
        :param population: population array 2d
        :param fitness: the fitness for each point in the population (a vector)
        :return: plt_to_img results
        """
        plt_to_img_results = []
        X_p, Y_p, Z_p = population[:, 0], population[:, 1], fitness
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
            X_last, Y_last, Z_last = self.last_XYZ
            X_diff, Y_diff, Z_diff = X_p - X_last, Y_p - Y_last, Z_p - Z_last
            resol_x, resol_y = self.anim_resolution
            steps = max(-int(np.mean(X_diff / resol_x + Y_diff / resol_y)), 3)

            X_diff, Y_diff, Z_diff = X_diff / steps, Y_diff / steps, Z_diff / steps
            for i in range(steps):
                self.collection_3d._offsets3d = X_last + i * X_diff, Y_last + i * Y_diff, Z_last + i * Z_diff
                plt_to_img_results.append(plt_to_img(self.fig_3d))

                # pause mode
                while self.pause_mode:
                    plt.pause(self.refresh_inter)
                plt.pause(self.refresh_inter)

        self.last_XYZ = (X_p, Y_p, Z_p)
        return plt_to_img_results


if __name__ == "__main__":
    print(bf.__all__)
    #########################################################################
    # problem = bf.Ackley(2)
    # problem = bf.StyblinskiTang(2)
    # # problem = bf.BukinN6()
    # lower = problem.min_search_range
    # upper = problem.max_search_range
    #
    # print(problem.get_global_optimum_solution(), problem.optimal_solution)
    # sfs = StochasticFractalSearch(lower, upper, 2, 50, .5, problem.get_func_val, 30, demo_3d=True)
    # sfs.optimize()
    #########################################################################
    # problem = bf.Easom()
    # lower = problem.min_search_range
    # upper = problem.max_search_range
    #
    # print(sample.get_global_optimum_solution(), sample.optimal_solution)
    # sfs = StochasticFractalSearch(lower, upper, 2, 50, .5, sample.get_func_val, 30, demo_3d=True, refresh_inter=.5, plot_step=.5)
    # sfs.optimize()
    #########################################################################
    # problem = bf.StyblinskiTang(10)
    # lower = problem.min_search_range
    # upper = problem.max_search_range
    # print("actual optimal value", problem.get_global_optimum_solution())
    # print("actual optimal point", problem.optimal_solution)
    # sfs = StochasticFractalSearch(lower, upper, 2, 50, .5, problem.get_func_val, 400, demo_3d=False)
    # sfs.optimize()
    # print("algorithm optimal value", sfs.best_fit)
    # print("algorithm optimal point", sfs.best_point)
