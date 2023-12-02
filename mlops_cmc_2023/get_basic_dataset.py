from sklearn.datasets import fetch_openml


def get_train_data():
    X, y = fetch_openml("mnist_784", return_X_y=True, parser="auto")
    return X.iloc[:6000], y[:6000]


def get_test_data():
    X, y = fetch_openml("mnist_784", return_X_y=True, parser="auto")
    return X.iloc[6000:], y[6000:]


if __name__ == "__main__":
    train_df = get_train_data()[0]
    train_df["target"] = get_train_data()[1]
    train_df.to_csv("train_data.csv")
    test_df = get_test_data()[0]
    test_df["target"] = get_test_data()[1]
    test_df.to_csv("test_data.csv")
