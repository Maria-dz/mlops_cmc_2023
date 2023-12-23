import torch


def main():
    images = torch.zeros((1, 785))
    model = torch.load("model_trained.pkl")
    traced_model = torch.jit.trace(model, images, check_trace=False)
    traced_model.to(torch.double)
    traced_model.save("../triton/model_repository/base_model/1/model.pt")


if __name__ == "__main__":
    main()
