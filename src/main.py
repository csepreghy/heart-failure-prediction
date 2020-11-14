# Standard library imports
import numpy as np
import pandas as pd

# Third party imports
from sklearn.metrics import accuracy_score

# Local imports
from mlp import MLP

if __name__ == '__main__':
    df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
    
    mlp = MLP()
    X_train, X_test, y_train, y_test = mlp.prepare_data(df)
    model = mlp.fit(X_train, X_test, y_train, y_test, epochs=50)
    
    y_pred = model.predict(X_test).flatten()
    y_pred = [round(value) for value in y_pred]
    print(f'y_pred = {y_pred}')

    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))