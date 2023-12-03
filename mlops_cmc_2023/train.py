import io
import pickle

import dvc.api
import hydra
import knn_classifier as knn_tools
import pandas as pd


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg):
    smth = dvc.api.read(cfg["train_data"]["path"])
    train_string = io.StringIO(smth)
    df = pd.read_csv(train_string, sep=",")

    y = df["target"]
    X = df.drop("target", axis=1)
    model = knn_tools.KNNClassifier(
        k=cfg["training"]["k"], metric=cfg["training"]["metric"]
    )
    model = model.fit(X, y)
    with open("model_trained.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
