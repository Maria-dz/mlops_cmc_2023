import io

import dvc.api
import hydra
import knn_classifier as knn_tools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg):
    smth = dvc.api.read(cfg["train_data"]["path"])
    train_string = io.StringIO(smth)
    df = pd.read_csv(train_string, sep=",", dtype=np.float32)

    y = np.array(df["target"])
    X = np.array(df.drop("target", axis=1))

    features_train = torch.from_numpy(X)
    target_train = torch.from_numpy(y).type(torch.LongTensor)
    batch_size = 256
    train = torch.utils.data.TensorDataset(features_train, target_train)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    train_losses = []

    model = knn_tools.Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0015)
    epochs = 15
    steps = 0
    print_every = 200
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
                """test_loss = 0
                accuracy = 0
                with torch.no_grad():
                model.eval()
                for images, labels in test_loader:
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))"""
                model.train()
                train_losses.append(running_loss / len(train_loader))
                # test_losses.append(test_loss/len(test_loader))
                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                )
                # "Test Loss: {:.3f}.. ".format(test_losses[-1]),
                # "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

    # model = knn_tools.KNNClassifier(
    #    k=cfg["training"]["k"], metric=cfg["training"]["metric"]
    # )
    # model = model.fit(X, y)
    # with open("model_trained.pkl", "wb") as f:
    #    pickle.dump(model, f)
    torch.save(model, "model_trained.pkl")


if __name__ == "__main__":
    main()
