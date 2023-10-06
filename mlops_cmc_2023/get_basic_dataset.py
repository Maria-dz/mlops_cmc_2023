from sklearn.datasets import fetch_openml


def get_train_data():
    X, y = fetch_openml("mnist_784", return_X_y=True, parser="auto")
    return X.iloc[:6000], y[:6000]


def get_test_data():
    X, y = fetch_openml("mnist_784", return_X_y=True, parser="auto")
    return X.iloc[6000:], y[6000:]
