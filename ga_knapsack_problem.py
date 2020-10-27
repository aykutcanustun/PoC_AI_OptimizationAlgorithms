import numpy as np


# Creates a random binary population with given shape:
def create_pop(pop_size, gene_num):
    population = np.random.randint(0, 2, gene_num*pop_size)
    population = population.reshape((pop_size, gene_num))
    return population


# Calculates fitness values of every individual in population:
def cal_fitness(value, weight, population, capacity):
    fitness = np.full(population.shape[0], -99999)
    for i in range(population.shape[0]):
        sum_value, sum_weight = 0, 0
        for j in range(population.shape[1]):
            sum_value += population[i][j] * value[j]
            sum_weight += population[i][j] * weight[j]
        if sum_weight <= capacity:
            fitness[i] = sum_value
        else:
            fitness[i] = 0
    return fitness


# Selects parents with elitism method:
def selection(population, fitness):    
    fitness = list(fitness)
    parents = np.full((2, population.shape[1]), -99999)
    for i in range(2):
        max_fitness_idx = np.where(fitness == np.max(fitness))[0][0]
        parents[i] = population[max_fitness_idx]
        fitness[max_fitness_idx] = -99999
    return parents


# Using selection function, creates 2 offsprings:
def crossover(parents):
    num_crossover = np.random.randint(1, 4)
    offsprings = parents.copy()
    for _ in range(num_crossover):
        cutpoint = np.random.randint(1, parents.shape[1])
        offsprings[0], offsprings[1] = np.append(offsprings[0][:cutpoint], offsprings[1][cutpoint:]), np.append(offsprings[1][:cutpoint], offsprings[0][cutpoint:])
    return offsprings


# Mutates offsprings that created in crossover fuction with 0,3 mutation probability:
def mutation(offsprings):
    mutation_probability = 0.3
    for i in range(offsprings.shape[1]):
        for j in offsprings:
            rand = np.random.rand()
            if rand < mutation_probability and j[i] == 1:
                j[i] = 0
            elif rand < mutation_probability and j[i] == 0:
                j[i] = 1
    return offsprings


# Finds optimum values with given iteration number:
def optimization(num_iteration):
    population = create_pop(8, items.shape[1])
    for _ in range(num_iteration):
        fitness = cal_fitness(items[0], items[1], population, capacity)
        parents = selection(population, fitness)
        offsprings = mutation(crossover(parents))
        population = np.vstack((population, offsprings))
        fitness = cal_fitness(items[0], items[1], population, capacity)
        new_population = np.full((8, items.shape[1]), -99999)
        for i in range(new_population.shape[0]):
            max_fitness_idx = np.where(fitness == np.max(fitness))[0][0]
            new_population[i] = population[max_fitness_idx]
            fitness[max_fitness_idx] = -99999
        population = new_population

    fitness = cal_fitness(items[0], items[1], population, capacity)
    optimum = population[np.where(fitness == np.max(fitness))[0][0]]
    value = np.sum(optimum * items[0])
    weight = np.sum(optimum * items[1])
    print("Optimised Population:\n", population)
    print("Optimised Solution After", num_iteration,"Iteration:")
    print("Optimum Chromosome:", optimum, "\nTotal Value:", value, "\nTotal Weight:", weight)
 

items = np.array([[135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240],  # Items profit values here.
                  [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]]) # Items weight values here.
capacity = 750 # Backpack weight capacity here.
optimization(500)
