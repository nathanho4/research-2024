import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import time

def load_data(file):
    data = np.genfromtxt(file, delimiter=';', dtype='str', skip_header=1)
    return data

def preprocess_data(data):
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def createModel(features_string, data):
    features = features_string[:36] + "1"
    kernel_string = features_string[37:39]
    gamma_string = features_string[39:]
     # Convert the feature string to a boolean mask
    feature_mask = np.array([int(char) for char in features]) == 1

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
    #print(f"Feature mask: {feature_mask}")

    # Select the features from the data using the mask
    selected_features = data[:, feature_mask]
    X, y = preprocess_data(selected_features)
    
    #print("Selected features shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    #print("Pre train")

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
    
    # 10 elements, each 40 random 1s and 0s
    currentpop = [''.join(np.random.choice(['0', '1'], size=40)) for _ in range(10)]
    testpop = ''.join(np.random.choice(['0', '1'], size=37)) + "011"
    testFitness = createModel(testpop, data)
    print(f"String: {testpop}, F1 Score: {testFitness}")
    # f1scores for each of the elements in currentpop
    fitness = []
    #for i in range(len(currentpop)):
        #f1score = createModel(currentpop[i], data)
        #fitness.append(f1score)

    #for i in range(len(currentpop)):
        #print(f"String: {currentpop[i]}, F1 Score: {fitness[i]}")
    #print("Fitness array:", fitness)

if __name__ == '__main__':
    main()