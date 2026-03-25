# %%
import cupy as cp
import numpy as np
import pandas as pd
import operator
import random
import gc
import warnings
import builtins
import operator
from deap import algorithms, base, creator, tools, gp


# 1. Force Python to delete any unlinked variables in system RAM
gc.collect()

# 2. Force CuPy to release all cached memory back to the GTX 1650
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

print("CPU and GPU memory caches cleared!")


# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate features and target in training set
X_train = train_df.drop(columns=['output'])
y_train = train_df['output']

X_test = test_df.drop(columns=['output'], errors='ignore')
y_test = test_df['output'] if 'output' in test_df.columns else None

# Transpose X_train so columns unpack correctly in the compiled GP equation
X_train_gpu = cp.asarray(X_train.values).T
y_train_gpu = cp.asarray(y_train.values)


# Define a function set for GP
pset = gp.PrimitiveSet("MAIN", X_train.shape[1])  # Number of features in the dataset

# Give DEAP's restricted compilation environment access to Python's core tools 
# so CuPy is allowed to compile GPU kernels on the fly
pset.context["__builtins__"] = builtins.__dict__

# Use explicit CuPy math functions instead of standard Python operators
# This prevents Python from getting confused between CPU and GPU math
pset.addPrimitive(cp.add, 2, name="add")
pset.addPrimitive(cp.subtract, 2, name="sub")
pset.addPrimitive(cp.multiply, 2, name="mul")
pset.addPrimitive(cp.negative, 1, name="neg")

pset.addPrimitive(cp.square, 1, name="square")
pset.addPrimitive(cp.sqrt, 1, name="sqrt")
pset.addPrimitive(cp.abs, 1, name= "abs")

# Force the random constants to be floats (CuPy handles floats better than raw integers)
pset.addEphemeralConstant("rand101", lambda: float(random.randint(-1, 1)))

# Define fitness and individual structure
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compile expression to evaluate individuals
toolbox.register("compile", gp.compile, pset=pset)


# Define the fitness function
def eval_gp(individual):
    func = toolbox.compile(expr=individual)
    
    try:
        # Use standard Python warnings instead of CuPy's missing errstate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Unpack the transposed GPU array. 
            # This passes all 95,000 rows for Feature 1 as arg1, Feature 2 as arg2, etc.
            raw_predictions = func(*X_train_gpu)
            
            # If the output is a single scalar (e.g., the tree was just a constant), 
            # we need to broadcast it to match the dataset size
            if cp.isscalar(raw_predictions):
                raw_predictions = cp.full(y_train_gpu.shape, raw_predictions)
            
            # Convert raw outputs to 1 or 0
            predictions = cp.where(raw_predictions > 0, 1, 0)
            
            # cp.mean returns a GPU scalar. Use .item() to send the float back to DEAP on the CPU
            accuracy = cp.mean(predictions == y_train_gpu).item()
            
            # Check for NaN (Not a Number) which ruins fitness tracking
            if cp.isnan(accuracy):
                return 0,
                
            return accuracy,
            
    except Exception as e:
        # If the generated equation crashes the GPU (e.g., massive memory overflow), 
        # kill the individual by giving it a fitness of 0.
        # Temporarily print the error so we can see what is breaking!
        print(f"GPU Error: {e}")
        return 0, 

toolbox.register("evaluate", eval_gp)

# Register tournament selection with tournsize parameter
toolbox.register("select", tools.selTournament, tournsize=3)

# Crossover, mutation, and limitation
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# This prevents "wide/dense" trees from crashing the parser
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=150))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=150))


# Evolution with elitism
def ea_with_elitism(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        # Select the next generation individuals with elitism
        elite_count = 2  # Elitism: keep the top 2 individuals
        elite = tools.selBest(population, elite_count)
        offspring = toolbox.select(population, len(population) - elite_count)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Combine elite individuals with offspring
        offspring.extend(elite)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

# Run the genetic programming algorithm
population = toolbox.population(n=1000)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Evolution with genetic programming
ea_with_elitism(population, toolbox, 0.9, 0.2, 70, stats=stats, halloffame=hof, verbose=True)

# Evaluate the best individual on the test set (if available)
best_ind = hof[0]
print("Best individual:", best_ind)
best_func = toolbox.compile(expr=best_ind)

# --- NEW VECTORIZED GPU TEST EVALUATION ---
# 1. Move test data to GPU and transpose it so columns unpack correctly
X_test_gpu = cp.asarray(X_test.values).T

# 2. Run the entire dataset through the GPU at once
raw_test_predictions = best_func(*X_test_gpu)

# 3. Handle the edge case where the equation is just a constant number
if cp.isscalar(raw_test_predictions):
    raw_test_predictions = cp.full(X_test.shape[0], raw_test_predictions)

# 4. Convert to 1s and 0s
test_predictions_gpu = cp.where(raw_test_predictions > 0, 1, 0)

# 5. Move the final predictions back to the CPU (NumPy) so you can save to CSV
test_predictions = cp.asnumpy(test_predictions_gpu)
# ------------------------------------------

if y_test is not None:
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
    test_accuracy = np.mean(test_predictions == y_test_np)
    print("Test Accuracy:", test_accuracy)
else:
    print("No target column in test set; predictions generated successfully.")
    #test_predictions = [1 if best_func(*x) > 0 else 0 for x in X_test.values]


submission_df = pd.DataFrame({
    'index': range(len(test_predictions)),  # Use test_predictions for index
    'output': test_predictions # Use test_predictions as output
})

# Check the shape of the submission DataFrame
print(submission_df.shape)

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

# Display the first few rows of the submission DataFrame
print(submission_df.head())