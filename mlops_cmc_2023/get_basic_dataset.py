from sklearn.datasets import fetch_openml

def get_data():
  mnist = fetch_openml("mnist_784")
  return mnist
