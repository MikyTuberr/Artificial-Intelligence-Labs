# Laboratory 1: Linear Regression 

## Overview
In this lab, we focus on linear regression, a fundamental technique in machine learning. Linear regression aims to find a linear relationship between input features and an outcome variable. We explore the concepts of linear regression, data splitting, cost functions, gradient descent, and standardization.

## Linear Regression
Linear regression involves finding a linear function to predict an outcome variable based on input features. The model parameters are optimized to minimize the error between predicted and actual values.

## Data Splitting
Datasets are divided into training and testing sets to train the model on one subset and evaluate its performance on another. This prevents overfitting and provides a measure of generalization.

## Cost Function
The cost function quantifies the difference between predicted and actual values. Mean squared error (MSE) is a common choice for regression problems.

## Gradient Descent
Gradient descent is an iterative optimization algorithm used to minimize the cost function. It adjusts model parameters in the direction of steepest descent of the cost function.

## Standardization
Standardization is a preprocessing step where input features are scaled to have a mean of 0 and a standard deviation of 1. This ensures that features are on a similar scale, facilitating the training process.

# Laboratory 2: Genetic Algorithm 

## Overview
In this lab, we delve into the application of genetic algorithms, a class of optimization algorithms inspired by the process of natural selection, to solve the Knapsack Problem. The Knapsack Problem involves selecting a combination of items with maximum value while not exceeding a given weight constraint.

## Genetic Algorithm
A genetic algorithm (GA) is a metaheuristic optimization technique inspired by the principles of natural selection and genetics. It involves creating a population of candidate solutions, evaluating their fitness, and iteratively evolving them to find the optimal solution.

## Creating Initial Population
The GA starts by generating an initial population of candidate solutions. Each solution represents a possible combination of items to be placed in the knapsack.

## Selection of Parents
Parents for the next generation are selected based on their fitness, which is determined by how well they satisfy the problem constraints and objectives. Common selection methods include roulette wheel selection, tournament selection, or rank-based selection.

## Crossover
Crossover is a genetic operator that combines genetic material from two parents to create offspring. In the context of the Knapsack Problem, crossover involves exchanging genetic information between solutions to explore new combinations of items.

## Mutation
Mutation introduces random changes in the offspring population to maintain genetic diversity and prevent premature convergence to suboptimal solutions. In the Knapsack Problem, mutation might involve adding or removing items from a solution.

## Updating Population
After generating offspring through crossover and mutation, the population is updated by replacing less fit individuals with the newly created offspring. This process continues for a predefined number of generations or until a termination condition is met.

## Solving the Knapsack Problem
The Knapsack Problem is a classic optimization problem where the goal is to maximize the total value of items selected while ensuring that the total weight does not exceed the capacity of the knapsack. Genetic algorithms offer a flexible and efficient approach to finding near-optimal solutions to this combinatorial optimization problem.

By implementing the genetic algorithm framework described above and adapting it to the Knapsack Problem, we can effectively explore the solution space and find feasible solutions that maximize the value of items packed in the knapsack within the weight constraint.
