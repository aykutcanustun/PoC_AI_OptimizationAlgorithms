import numpy as np


def create_pop(pop_size=10, gene_num=10):
    population = np.random.randint(0, 2, gene_num*pop_size)
    population = population.reshape((pop_size, gene_num))
    total = 0
    for i in population:
        if np.sum(i) > total:
            total = np.sum(i)
            optimum = i

    print(f"Starting Population::\n{population}")
    print(f"Unoptimised Best Value::{optimum}:[{np.sum(optimum)}]\n")
    return population


def cal_fitness(population):
    totals = np.sum(population, axis=1)
    indices = np.argsort(totals)
    fitness = np.array([])
    for i in range(len(totals)):
        if i != 0:
            fitness = np.hstack(
                (fitness, totals[i]/np.sum(totals)+fitness[i-1], 6))
        else:
            fitness = np.hstack((fitness, totals[i]/np.sum(totals), 6))
    return fitness, indices


def roulette_wheel(fitness):
    rand = np.random.random_sample()
    for i in fitness:
        if rand <= i:
            return i


def select_parents(population, fitness, indices):
    parents = np.array([], dtype=int)
    rand = [roulette_wheel(fitness), roulette_wheel(fitness)]
    while rand[0] == rand[1]:
        rand = [roulette_wheel(fitness), roulette_wheel(fitness)]
    for i in range(2):
        parents = np.append(parents, np.where(fitness == rand[i])[0])
    return population[indices[parents[0]]], parents[0], population[indices[parents[1]]], parents[1]


def crossover(p1, p2, gene_num=10):
    num_crossover = np.random.randint(1, 4)
    c1, c2 = p1, p2
    for _ in range(num_crossover):
        cutpoint = np.random.randint(1, gene_num)
        c1, c2 = np.append(c1[:cutpoint], c2[cutpoint:]), np.append(
            c2[:cutpoint], c1[cutpoint:])
    return c1, c2


def mutation(c1, c2, gene_num=10):
    mutation_probability = 0.3
    for i in range(gene_num):
        for j in c1, c2:
            rand = np.random.rand()
            if rand < mutation_probability and j[i] == 1:
                j[i] = 0
            elif rand < mutation_probability and j[i] == 0:
                j[i] = 1
    return c1, c2


def iteration(num_iteration=1000):
    population = create_pop()
    for _ in range(1, num_iteration+1):

        fitness, indices = cal_fitness(population=population)
        p1, p1_indice, p2, p2_indice = select_parents(
            population=population, fitness=fitness, indices=indices)
        c1, c2 = crossover(p1=p1, p2=p2)
        c1, c2 = mutation(c1=c1, c2=c2)
        p1_total, p2_total = np.sum(p1), np.sum(p2)
        c1_total, c2_total = np.sum(c1), np.sum(c2)

        if c1_total > c2_total:  # When child1 is better than child2:
            if p1_total > p2_total:
                if c1_total > p1_total or c1_total == p1_total:
                    population[indices[p1_indice]] = c1
                    if c2_total > p2_total or c2_total == p2_total:
                        population[indices[p2_indice]] = c2
                elif c1_total > p2_total or c1_total == p2_total:
                    population[indices[p2_indice]] = c1

            elif p1_total < p2_total:
                if c1_total > p2_total or c1_total == p2_total:
                    population[indices[p2_indice]] = c1
                    if c2_total > p1_total or c2_total == p1_total:
                        population[indices[p1_indice]] = c2
                elif c1_total > p1_total or c1_total == p1_total:
                    population[indices[p1_indice]] = c1

            elif p1_total == p2_total:
                if c2_total > p2_total or c2_total == p2_total:
                    population[indices[p1_indice]] = c1
                    population[indices[p2_indice]] = c2
                elif c1_total > p1_total or c1_total == p1_total:
                    population[indices[p1_indice]] = c1

        elif c1_total < c2_total:  # When child2 is better than child1:
            if p1_total > p2_total:
                if c2_total > p1_total or c2_total == p1_total:
                    population[indices[p1_indice]] = c2
                    if c1_total > p2_total or c1_total == p2_total:
                        population[indices[p2_indice]] = c1
                elif c2_total > p2_total or c1_total == p2_total:
                    population[indices[p2_indice]] = c2

            elif p1_total < p2_total:
                if c2_total > p2_total or c2_total == p2_total:
                    population[indices[p2_indice]] = c2
                    if c1_total > p1_total or c1_total == p1_total:
                        population[indices[p1_indice]] = c1
                elif c2_total > p1_total or c2_total == p1_total:
                    population[indices[p1_indice]] = c2

            elif p1_total == p2_total:
                if c1_total > p1_total or c1_total == p1_total:
                    population[indices[p1_indice]] = c1
                    population[indices[p2_indice]] = c2
                elif c2_total > p2_total or c2_total == p2_total:
                    population[indices[p2_indice]] = c2

        elif c1_total == c2_total:  # When child1 and child2 both equally good:
            if p1_total > p2_total:
                if c1_total > p1_total or c1_total == p1_total:
                    population[indices[p1_indice]] = c1
                    population[indices[p2_indice]] = c2
                elif c1_total > p2_total or c1_total == p2_total:
                    population[indices[p2_indice]] = c1

            elif p1_total < p2_total:
                if c2_total > p2_total or c2_total == p2_total:
                    population[indices[p1_indice]] = c1
                    population[indices[p2_indice]] = c2
                elif c1_total > p1_total or c1_total == p1_total:
                    population[indices[p1_indice]] = c1

            elif p1_total == p2_total:
                if c1_total > p1_total or c1_total == p1_total:
                    population[indices[p1_indice]] = c1
                    population[indices[p2_indice]] = c2

    total = 0
    for i in population:
        if np.sum(i) > total:
            total = np.sum(i)
            optimum = i

    print(f"End Population After {num_iteration} Iteration::\n{population}")
    print(f"Optimised Best Value::{optimum}:[{np.sum(optimum)}]")
    return optimum


optimum = iteration()
