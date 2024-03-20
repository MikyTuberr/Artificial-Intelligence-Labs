from itertools import compress
import random
import time
import matplotlib.pyplot as plt
from typing import List, Tuple

from data import *

def initial_population(individual_size: int, population_size: int) -> List[List[bool]]:
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items: dict, knapsack_max_capacity: int, individual: List[bool]) -> int:
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items: dict, knapsack_max_capacity: int, population: List[List[bool]]) -> Tuple[List[bool], int]:
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def roulette_wheel_selection(items: dict, population: List[List[bool]], knapsack_max_capacity: int, n_selection: int) -> List[List[bool]]:
    fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    sum_of_fitnesses = sum(fitnesses)
    probabilities = [fit / sum_of_fitnesses for fit in fitnesses]

    return random.choices(population, weights=probabilities, k=n_selection)

def single_point_crossover(parent1: List[bool], parent2: List[bool]) -> Tuple[List[bool], List[bool]]:
    crossover_point = random.randint(0, len(parent1) - 1)

    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2

def make_children(parents: List[List[bool]]) -> List[List[bool]]:
    children = [single_point_crossover(random.choice(parents), random.choice(parents)) for _ in range(0, len(parents), 2)]
    return [child for sublist in children for child in sublist]

def mutation(individual: List[bool], p_mut: float) -> List[bool]:
    for i in range(len(individual)):
        if random.random() < p_mut:
            individual[i] = not individual[i]
    return individual

def mutate_generation(generation: List[List[bool]], p_mut: float) -> List[List[bool]]:
    return [mutation(individual, p_mut) for individual in generation]

def update_population(n_elite: int, population: List[List[bool]], new_generation: List[List[bool]], items: dict, knapsack_max_capacity: int) -> List[List[bool]]:
    elite = sorted(population, key=lambda individual: fitness(items, knapsack_max_capacity, individual), reverse=True)[:n_elite]
    population = elite + new_generation + population[:len(new_generation) - n_elite]
    return population

items, knapsack_max_capacity = get_big()

population_size = 100
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    parents = roulette_wheel_selection(items, population, knapsack_max_capacity, n_selection)
    new_generation = make_children(parents)
    new_generation = mutate_generation(new_generation, 0.1)
    population = update_population(n_elite, population, new_generation, items, knapsack_max_capacity)

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
