import numpy as np
import pandas as pd


# Creates a random population with given shape:
def create_pop(pop_size, gene_num):
    population = np.full((pop_size, gene_num), -99999)
    arr = np.arange(gene_num)
    for i in range(pop_size):
        np.random.shuffle(arr)
        population[i] = arr
    return population


# Calculates fitness values of every individual in population:
def cal_fitness(distances, population):
    fitness = np.full(population.shape[0], -99999)
    for i in range(population.shape[0]):
        distance = 0
        for j in range(population.shape[1] - 1):
            distance += distances.loc[population[i, j], population[i, j+1]]
        fitness[i] = 100000 / distance
    indices = np.argsort(fitness)
    return fitness, indices


# Selects parents with elitism method:
def selection(population, fitness):    
    fitness = list(fitness)
    parents = np.full((2, population.shape[1]), -99999)
    parent_indices = np.full(2, -99999, dtype=int)
    for i in range(2):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i] = population[max_fitness_idx[0][0]]
        parent_indices[i] = max_fitness_idx[0][0]
        fitness[max_fitness_idx[0][0]] = -99999
    return parents, parent_indices
        
   
# Creates 2 offsprings with ordered crossover method:
def crossover(parents):
    offsprings = np.full((2, parents.shape[1]), -99999)
    gene0, gene1 = int(np.random.rand() * parents.shape[1]), int(np.random.rand() * parents.shape[1])
    startGene, endGene = min(gene0, gene1), max(gene0, gene1)
    for i in range(2):
        offsprings[i][startGene:endGene+1] = parents[i][startGene:endGene+1]
        for j in range(offsprings.shape[1]):
            for k in parents[0]:
                if k not in offsprings[i] and i==1 and offsprings[i][j] == -99999:
                    offsprings[i][j] = k
            for k in parents[1]:
                if k not in offsprings[i] and i==0 and offsprings[i][j] == -99999:
                    offsprings[i][j] = k
    return offsprings


# Mutates offsprings with 0.2 mutation probability and swap mutation method:
def mutation(offsprings):
    mutation_probability = 0.2
    for i in range(offsprings.shape[1]):
        for j in range(2):
            if np.random.rand() < mutation_probability:
                swap_gene = int(np.random.rand() * offsprings.shape[1])
                offsprings[j][i], offsprings[j][swap_gene] = offsprings[j][swap_gene], offsprings[j][i]
    return offsprings


distances=pd.DataFrame([[0, 29, 82, 46, 68, 52, 72, 42, 51, 55, 29, 74, 23, 72, 46],
                        [29, 0, 55, 46, 42, 43, 43, 23, 23, 31, 41, 51, 11, 52, 21],
                        [82, 55, 0, 69, 46, 55, 23, 43, 41, 29, 79, 21, 64, 31, 51],
                        [46, 46, 68, 0, 82, 15, 72, 31, 62, 42, 21, 51, 52, 43, 64],
                        [68, 42, 46, 82, 0, 74, 23, 52, 21, 46, 82, 58, 46, 65, 23],
                        [52, 43, 55, 15, 74, 0, 61, 23, 55, 31, 33, 37, 51, 29, 59],
                        [72, 43, 23, 72, 23, 61, 0, 42, 23, 31, 77, 37, 51, 46, 33],
                        [42, 23, 43, 31, 52, 23, 42, 0, 33, 15, 37, 33, 33, 31, 37],
                        [51, 23, 41, 62, 21, 55, 23, 33, 0, 29, 62, 46, 29, 51, 11],
                        [55, 31, 29, 42, 46, 31, 31, 15, 29, 0, 51, 21, 41, 23, 37],
                        [29, 41, 79, 21, 82, 33, 77, 37, 62, 51, 0, 65, 42, 59, 61],
                        [74, 51, 21, 51, 58, 37, 37, 33, 46, 21, 65, 0, 61, 11, 55],
                        [23, 11, 64, 51, 46, 51, 51, 33, 29, 41, 42, 61, 0, 62, 23],
                        [72, 52, 31, 43, 65, 29, 46, 31, 51, 23, 59, 11, 62, 0, 59],
                        [46, 21, 51, 64, 23, 59, 33, 37, 11, 37, 61, 55, 23, 59, 0]])


# Main loop, finds optimum values with given iteration number:
population = create_pop(8, distances.shape[1])
num_iteration = 1000
for _ in range(num_iteration):
    fitness, indices = cal_fitness(distances, population)
    parents, parent_indices = selection(population, fitness)
    parent_fitness = cal_fitness(distances, parents)[0]
    offsprings = crossover(parents)
    offsprings = mutation(offsprings)
    offspring_fitness = cal_fitness(distances, offsprings)[0]
    
    # When offspring1 is better than offspring2:
    if offspring_fitness[0] > offspring_fitness[1]:
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

    # When offspring2 is better than offspring1:
    elif offspring_fitness[0] < offspring_fitness[1]:
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

    # When offspring1 and offspring2 both equally good:
    elif offspring_fitness[0] == offspring_fitness[1]:
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

fitness = cal_fitness(distances, population)[0]
optimum = population[np.where(fitness == np.max(fitness))[0][0]]
distance = 0
for i in range(len(optimum)-1):
    distance += distances.loc[optimum[i], optimum[i+1]]
print("Optimised Population:\n", population)
print("Optimised Solution After", num_iteration, "Iteration:")
print("Optimum Chromosome:", optimum, "\nDistance:", distance)
