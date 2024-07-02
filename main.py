import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time

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
    elif kernel_string == "01":
        kernel_type = "poly"
    elif kernel_string == "10":
        kernel_type = "sigmoid"
    else:
        kernel_type = "rbf"

    if gamma_string == "0":
        gamma_type = "scale"
    else:
        gamma_type = "auto"
    
    print(f"Kernel type: {kernel_type}, Gamma type: {gamma_type}")

    # Select the features from the data using the mask
    selected_features = data[:, feature_mask]
    X, y = preprocess_data(selected_features)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    

    model = SVC(kernel=kernel_type, gamma=gamma_type)

    start_time = time.time()
    model.fit(X_train, y_train)
    #print("Post train")
    print(f"Training time: {time.time() - start_time} seconds")

    y_pred = model.predict(X_test)
    f1score = metrics.f1_score(y_test, y_pred, average='weighted')
    
    return f1score

def main():
    data = load_data('data.csv')
    testing = True
    
    #testing SVC model with poly kernel type
    if testing:
        testpop = ''.join(np.random.choice(['0', '1'], size=37)) + "011"
        testFitness = createModel(testpop, data)
        print(f"String: {testpop}, F1 Score: {testFitness}")

    else:
        # 10 elements, each 40 random 1s and 0s
        currentpop = [''.join(np.random.choice(['0', '1'], size=40)) for _ in range(10)]
        
        # f1scores for each of the elements in currentpop
        fitness = []
        for i in range(len(currentpop)):
            f1score = createModel(currentpop[i], data)
            fitness.append(f1score)
            print(f"String: {currentpop[i]}, F1 Score: {f1score}")

        #for i in range(len(currentpop)):
            #print(f"String: {currentpop[i]}, F1 Score: {fitness[i]}")

if __name__ == '__main__':
    main()