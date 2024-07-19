import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import pygad

def load_data(file):
    #reads the data from the csv file and returns it as a numpy array (ndarray)
    data = np.genfromtxt(file, delimiter=';', dtype='str', skip_header=1)
    return data


def preprocess_data(data):
    #splits the data into features (X) and labels (y)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def createModel(features_string, data):
    #extracts the first 36 characters from the features_string and appends a '1' 
    #to the end so that it includes the target column (the last column in the data, graduate/enrolled/dropout)
    features = features_string[:36] + "1"

    # Extract the kernel and gamma strings from the features string
    kernel_string = features_string[37:39]
    gamma_string = features_string[39:]

    # Convert the feature string to a boolean mask
    feature_mask = np.array([int(char) for char in features]) == 1

    #based on the kernel_string and gamma_string, set the kernel and gamma parameters for the SVC model
    kernel_type = ""
    gamma_type = ""
    if kernel_string == "00":
        kernel_type = "linear"
    #elif kernel_string == "01":
        #kernel_type = "poly"
    elif kernel_string == "10":
        kernel_type = "sigmoid"
    else:
        kernel_type = "rbf"

    if gamma_string == "0":
        gamma_type = "scale"
    else:
        gamma_type = "auto"
    
    #print(f"\nKernel type: {kernel_type}, Gamma type: {gamma_type}")

    # Select the features from the data using the mask
    selected_features = data[:, feature_mask]
    X, y = preprocess_data(selected_features)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)
    

    model = SVC(kernel=kernel_type, gamma=gamma_type)

    #start_time = time.time()
    model.fit(X_train, y_train)
    #print(f"Training time: {time.time() - start_time} seconds")

    y_pred = model.predict(X_test)
    f1score = metrics.f1_score(y_test, y_pred, average='weighted')
    
    return f1score


#there might be a better way to do this, but the function below generates 
#a fitness function that can be used by the genetic algorithm in order to work around
#the PyGAD fitness function requirements and the createModel function parameters
def fitness_func_creator(data):
    def fitness_func(ga_instance, solution, solution_idx):
        features_string = ''.join(map(str, map(int, solution)))
        return createModel(features_string, data)
    return fitness_func


# Function to be called after each generation
def on_generation(ga_instance):
    best_fitness = max(ga_instance.last_generation_fitness)
    print(f"Generation {ga_instance.generations_completed}: Best Fitness Value: {best_fitness}")


def genetic_algorithm(data, currentpop):
    num_generations = 1000
    #the parameter below sets the number of parents that will be selected for mating
    num_parents_mating = 2
    #number of solutions per population
    sol_per_pop = len(currentpop)
    #number of characters in the solution string
    num_genes = len(currentpop[0])
    mutation_probability = 0.0025

    initial_population = np.array([[int(char) for char in individual] for individual in currentpop])
    
    fitness_func = fitness_func_creator(data)
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           mutation_probability=mutation_probability,
                           initial_population=initial_population,
                           gene_space=[0, 1], # Ensures the string contains only 1s and 0s
                           on_generation=on_generation)
    
    ga_instance.run()
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best solution: {''.join(map(str, map(int, solution)))}")
    print(f"Best solution fitness: {solution_fitness}")


def main():
    data = load_data('data.csv')
    testing = False
    
    #testing SVC model with poly kernel type
    if testing:
        testpop = ''.join(np.random.choice([ '0', '1'], size=37)) + "011"
        testFitness = createModel(testpop, data)
        print(f"String: {testpop}, F1 Score: {testFitness}")

    else:
        # 10 elements, each 40 random 1s and 0s
        currentpop = [''.join(np.random.choice(['0', '1'], size=40)) for _ in range(100)]

        genetic_algorithm(data, currentpop)


if __name__ == '__main__':
    main()