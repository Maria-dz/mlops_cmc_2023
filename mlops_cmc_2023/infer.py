import io

import dvc.api
import hydra
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg):
    batch_size = cfg["training"]["batch_size"]
    smth = dvc.api.read(cfg["test_data"]["path"])
    train_string = io.StringIO(smth)
    df = pd.read_csv(train_string, sep=",", dtype=np.float32)
    y = np.array(df["target"])
    X = np.array(df.drop("target", axis=1))
    features_test = torch.from_numpy(X)
    target_test = torch.from_numpy(y).type(torch.LongTensor)
    test = torch.utils.data.TensorDataset(features_test, target_test)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    model = torch.load("model_trained.pkl")
    accuracy = 0
    y_pred = []
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            log_ps = model(images)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            y_pred += list(top_class.detach().numpy().reshape(-1))
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    res_dict = {"result": list(y_pred)}
    df = pd.DataFrame(res_dict)
    df.to_csv("predicted_labels.csv")
    print("accuracy: ", accuracy.detach().item() / len(test_loader))
    print("Results are saved to 'predicted_labels.csv'")


if __name__ == "__main__":
    main()
