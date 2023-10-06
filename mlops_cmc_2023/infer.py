import pickle

import get_basic_dataset as data
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    X_test, y_test = data.get_test_data()
    with open("model_trained.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    res_dict = {"result": list(y_pred)}
    df = pd.DataFrame(res_dict)
    df.to_csv("predicted_labels.csv")

    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("Results are saved to 'predicted_labels.csv'")
