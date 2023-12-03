import io
import pickle

import dvc.api
import hydra
import pandas as pd
from sklearn.metrics import accuracy_score


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg):
    smth = dvc.api.read(cfg["test_data"]["path"])
    train_string = io.StringIO(smth)
    df = pd.read_csv(train_string, sep=",")
    y = df["target"]
    X = df.drop("target", axis=1)
    with open("model_trained.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    res_dict = {"result": list(y_pred)}
    df = pd.DataFrame(res_dict)
    df.to_csv("predicted_labels.csv")

    print("accuracy: ", accuracy_score(y, y_pred))
    print("Results are saved to 'predicted_labels.csv'")


if __name__ == "__main__":
    main()
