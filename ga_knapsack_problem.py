import numpy as np


# Creates a random binary population with given shape:
def create_pop(pop_size, gene_num):
    population = np.random.randint(0, 2, gene_num*pop_size)
    population = population.reshape((pop_size, gene_num))
    return population


# Calculates fitness values of every individual in population:
def cal_fitness(value, weight, population, capacity):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        sum_value, sum_weight = 0, 0
        for j in range(population.shape[1]):
            sum_value += population[i][j] * value[j]
            sum_weight += population[i][j] * weight[j]
        if sum_weight <= capacity:
            fitness[i] = sum_value
        else:
            fitness[i] = 0
    indices = np.argsort(fitness)
    return fitness, indices


# Selects parents with elitism method:
def selection(fitness, population):
    fitness = list(fitness)
    parents = np.empty((2, population.shape[1]))
    parent_indices = np.empty(2, dtype=int)
    for i in range(2):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i] = population[max_fitness_idx[0][0]]
        parent_indices[i] = max_fitness_idx[0][0]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents, parent_indices


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


# Main loop, finds optimum values with given iteration number:
# Item No:       | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
items = np.array([[92, 57, 49, 68, 60, 43, 67, 84, 87, 72],  # Items profit values here.
                  [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]]) # Items weight values here.
capacity = 165 # Backpack weight capacity here.
num_iteration = 1000
population = create_pop(8, items.shape[1])
for _ in range(num_iteration):
    fitness, indices = cal_fitness(items[0], items[1], population, capacity)
    parents, parent_indices = selection(fitness, population)
    parent_fitness = cal_fitness(items[0], items[1], parents, capacity)[0]
    offsprings = crossover(parents)
    offsprings = mutation(offsprings)
    offspring_fitness = cal_fitness(items[0], items[1], offsprings, capacity)[0]

    if offspring_fitness[0] > offspring_fitness[1]:  # When offspring1 is better than offspring2:
        if parent_fitness[0] > parent_fitness[1]:
            if offspring_fitness[0] > parent_fitness[0] or offspring_fitness[0] == parent_fitness[0]:
                population[indices[parent_indices[0]]] = offsprings[0]
                if offspring_fitness[1] > parent_fitness[1] or offspring_fitness[1] == parent_fitness[1]:
                    population[indices[parent_indices[1]]] = offsprings[1]
            elif offspring_fitness[0] > parent_fitness[1] or offspring_fitness[0] == parent_fitness[1]:
                population[indices[parent_indices[1]]] = offsprings[0]
        elif parent_fitness[0] < parent_fitness[1]:
            if offspring_fitness[0] > parent_fitness[1] or offspring_fitness[0] == parent_fitness[1]:
                population[indices[parent_indices[1]]] = offsprings[0]
                if offspring_fitness[1] > parent_fitness[0] or offspring_fitness[1] == parent_fitness[0]:
                    population[indices[parent_indices[0]]] = offsprings[1]
            elif offspring_fitness[0] > parent_fitness[0] or offspring_fitness[0] == parent_fitness[0]:
                population[indices[parent_indices[0]]] = offsprings[0]
        elif parent_fitness[0] == parent_fitness[1]:
            if offspring_fitness[1] > parent_fitness[1] or offspring_fitness[1] == parent_fitness[1]:
                population[indices[parent_indices[0]]] = offsprings[0]
                population[indices[parent_indices[1]]] = offsprings[1]
            elif offspring_fitness[0] > parent_fitness[0] or offspring_fitness[0] == parent_fitness[0]:
                population[indices[parent_indices[0]]] = offsprings[0]

    elif offspring_fitness[0] < offspring_fitness[1]:  # When offspring2 is better than offspring1:
        if parent_fitness[0] > parent_fitness[1]:
            if offspring_fitness[1] > parent_fitness[0] or offspring_fitness[1] == parent_fitness[0]:
                population[indices[parent_indices[0]]] = offsprings[1]
                if offspring_fitness[0] > parent_fitness[1] or offspring_fitness[0] == parent_fitness[1]:
                    population[indices[parent_indices[1]]] = offsprings[0]
            elif offspring_fitness[1] > parent_fitness[1] or offspring_fitness[0] == parent_fitness[1]:
                population[indices[parent_indices[1]]] = offsprings[1]
        elif parent_fitness[0] < parent_fitness[1]:
            if offspring_fitness[1] > parent_fitness[1] or offspring_fitness[1] == parent_fitness[1]:
                population[indices[parent_indices[1]]] = offsprings[1]
                if offspring_fitness[0] > parent_fitness[0] or offspring_fitness[0] == parent_fitness[0]:
                    population[indices[parent_indices[0]]] = offsprings[0]
            elif offspring_fitness[1] > parent_fitness[0] or offspring_fitness[1] == parent_fitness[0]:
                population[indices[parent_indices[0]]] = offsprings[1]
        elif parent_fitness[0] == parent_fitness[1]:
            if offspring_fitness[0] > parent_fitness[0] or offspring_fitness[0] == parent_fitness[0]:
                population[indices[parent_indices[0]]] = offsprings[0]
                population[indices[parent_indices[1]]] = offsprings[1]
            elif offspring_fitness[1] > parent_fitness[1] or offspring_fitness[1] == parent_fitness[1]:
                population[indices[parent_indices[1]]] = offsprings[1]

    elif offspring_fitness[0] == offspring_fitness[1]:  # When offspring1 and offspring2 both equally good:
        if parent_fitness[0] > parent_fitness[1]:
            if offspring_fitness[0] > parent_fitness[0] or offspring_fitness[0] == parent_fitness[0]:
                population[indices[parent_indices[0]]] = offsprings[0]
                population[indices[parent_indices[1]]] = offsprings[1]
            elif offspring_fitness[0] > parent_fitness[1] or offspring_fitness[0] == parent_fitness[1]:
                population[indices[parent_indices[1]]] = offsprings[0]
        elif parent_fitness[0] < parent_fitness[1]:
            if offspring_fitness[1] > parent_fitness[1] or offspring_fitness[1] == parent_fitness[1]:
                population[indices[parent_indices[0]]] = offsprings[0]
                population[indices[parent_indices[1]]] = offsprings[1]
            elif offspring_fitness[0] > parent_fitness[0] or offspring_fitness[0] == parent_fitness[0]:
                population[indices[parent_indices[0]]] = offsprings[0]
        elif parent_fitness[0] == parent_fitness[1]:
            if offspring_fitness[0] > parent_fitness[0] or offspring_fitness[0] == parent_fitness[0]:
                population[indices[parent_indices[0]]] = offsprings[0]
                population[indices[parent_indices[1]]] = offsprings[1]

fitness = cal_fitness(items[0], items[1], population, capacity)[0]
optimum = population[np.where(fitness == np.max(fitness))[0][0]]
value = np.sum(optimum * items[0])
weight = np.sum(optimum * items[1])
print("Optimised Population:\n", population)
print("Optimised Solution After", num_iteration,"Iteration:")
print("Optimum Chromosome:", optimum, "\nTotal Value:", value, "\nTotal Weight:", weight)
