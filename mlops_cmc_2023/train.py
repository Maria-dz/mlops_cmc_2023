import pickle

import get_basic_dataset as data
import knn_classifier as knn_tools

if __name__ == "__main__":
    X, y = data.get_train_data()
    model = knn_tools.KNNClassifier(k=500, strategy="auto", metric="cosine")
    model = model.fit(X, y)
    with open("model_trained.pkl", "wb") as f:
        pickle.dump(model, f)
