import io
import subprocess
from datetime import datetime
from pathlib import Path

import dvc.api
import hydra
import knn_classifier as knn_tools
import matplotlib.pyplot as plt
import mlflow
import mlflow.onnx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg):
    mlflow.set_tracking_uri(cfg["mlflow_config"]["MLFLOW_Tracking_Uri"])
    experiment_id = mlflow.create_experiment(
        "MNIST_" + datetime.now().strftime("%m-%d %H:%M:%S"),
        artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        tags={"version": "v3", "priority": "P1"},
    )

    data = dvc.api.read(cfg["train_data"]["path"])
    train_string = io.StringIO(data)
    df = pd.read_csv(train_string, sep=",", dtype=np.float32)

    mlflow.start_run(
        experiment_id=experiment_id,
        run_name="MNIST" + f'{datetime.now().strftime("%m-%d %H:%M:%S")}',
    )

    y = np.array(df["target"])
    X = np.array(df.drop("target", axis=1))

    features_train = torch.from_numpy(X)
    target_train = torch.from_numpy(y).type(torch.LongTensor)
    batch_size = cfg["training"]["batch_size"]
    train = torch.utils.data.TensorDataset(features_train, target_train)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    smth = dvc.api.read(cfg["test_data"]["path"])
    train_string = io.StringIO(smth)
    df = pd.read_csv(train_string, sep=",", dtype=np.float32)
    y = np.array(df["target"])
    X = np.array(df.drop("target", axis=1))
    features_test = torch.from_numpy(X)
    target_test = torch.from_numpy(y).type(torch.LongTensor)
    test = torch.utils.data.TensorDataset(features_test, target_test)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    train_losses = []
    test_losses = []
    model = knn_tools.Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    epochs = cfg["training"]["num_epochs"]
    steps = 0
    print_every = cfg["training"]["print_every"]
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            steps += 1
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    model.eval()
                    for images, labels in test_loader:
                        log_ps = model(images)
                        test_loss += criterion(log_ps, labels)
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                model.train()
                train_losses.append(running_loss / len(train_loader))
                test_losses.append(test_loss / len(test_loader))
                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {} ".format(train_losses[-1]),
                    "Test Loss: {} ".format(test_losses[-1]),
                    "Test Accuracy: {}".format(accuracy / len(test_loader)),
                )
    plt.figure(figsize=(9, 9))
    plt.plot(train_losses, label="Training loss")
    plt.savefig("Training_loss.png")
    plt.figure(figsize=(9, 9))
    mlflow.log_artifact("Training_loss.png")
    plt.plot(test_losses, label="Test loss")
    plt.savefig("Test_loss.png")
    mlflow.log_artifact("Test_loss.png")

    mlflow.log_metric("train_loss", np.mean(train_losses))
    mlflow.log_metric("test_loss", np.mean(test_losses))

    torch.save(model, "model_trained.pkl")

    commit_id = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("utf-8")
    )
    mlflow.log_params(cfg["training"])
    mlflow.log_param("git_commit_id", commit_id)
    mlflow.end_run()


if __name__ == "__main__":
    main()
