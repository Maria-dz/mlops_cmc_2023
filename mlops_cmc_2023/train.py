import io
import pickle

import dvc.api
import knn_classifier as knn_tools
import pandas as pd

if __name__ == "__main__":
    smth = dvc.api.read("../data/test_data.csv")
    train_string = io.StringIO(smth)
    df = pd.read_csv(train_string, sep=",")

    y = df["target"]
    X = df.drop("target", axis=1)
    model = knn_tools.KNNClassifier(k=500, strategy="auto", metric="cosine")
    model = model.fit(X, y)
    with open("model_trained.pkl", "wb") as f:
        pickle.dump(model, f)
