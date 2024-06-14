import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(file):
    data = np.genfromtxt(file, delimiter=';', dtype='str', skip_header=1)
    
    X = data[:, :-1]
    y = data[:, -1]
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Identify categorical columns
    categorical_cols = np.where(~np.any(np.char.isdigit(X), axis=0))[0]
    
    # Pipeline for preprocessing using OneHotEncoder to account for the categorical columns
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    print(X_processed)
    print(y)

    return X_processed, y

def createModel(file):
    X, y = preprocess_data(file)
    
    model = SVC(kernel='linear')
    model.fit(X, y)
    
    y_pred = model.predict(X)
    f1score = metrics.f1_score(y, y_pred, average='weighted')
    
    return f1score

def main():
    print(createModel('data.csv'))

if __name__ == '__main__':
    main()