import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from activations import ReLU, Tanh, Softmax, Sigmoid
from conv import Conv2d
from modules import Module, Linear, CrossEntropy, Flatten
from optims import Adam, SGD, RMSProp, Adagrad


class MyNet(Module):
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, y_hat, y):
        return self.cost(y_hat, y)

    def backward(self):
        grad = self.cost.backward()
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)

    def parameters(self):
        res = []
        for layer in self.layers:
            res.extend(layer.parameters())
        return res


def train(model, lr, nb_epoch, loaders):
    optimizer = Adam(model.parameters(), lr=lr)
    # optimizer = RMSProp(model.parameters(), lr=lr, alpha=0.9)
    # optimizer = Adagrad(model.parameters(), lr=lr)

    print("Start training...")
    for epoch in range(nb_epoch):
        print(f"--- Epoch {epoch + 1}/{nb_epoch}")

        for phase in ["Train", "Test"]:
            accuracy = 0
            running_loss = 0.0
            num_inputs = 0
            for (inputs, targets) in tqdm(loaders[phase]):
                # for (inputs, targets) in loaders[phase]:
                num_inputs += inputs.shape[0]
                predicted = model(inputs)
                running_loss += model.loss(predicted, targets).sum()

                if phase == "Train":
                    optimizer.zero_grad()
                    model.backward()
                    optimizer.step()

                accuracy += (predicted.argmax(axis=1) == targets.argmax(axis=1)).sum()
            print(f'Phase {phase}. Loss: {running_loss / num_inputs}. Accuracy: {accuracy / num_inputs}')


def load_minibatches(batch_size=64):
    tsfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('.', train=True, download=True, transform=tsfms)
    test_set = datasets.MNIST('.', train=False, download=True, transform=tsfms)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=4, drop_last=True)

    # return {"Train": train_loader, "Test": test_loader}
    def transform_loader(loader):
        # inputs = inputs.reshape(batch_size, -1).numpy()
        return [(inputs.numpy(), np.eye(10)[targets.numpy()]) for (inputs, targets) in loader]

    return {"Train": transform_loader(train_loader), "Test": transform_loader(test_loader)}


def main():
    ker = 3
    pad = 1
    out_chan = 2
    stride = 1
    net = MyNet([
        Conv2d(in_channels=1, out_channels=out_chan, kernel_size=ker, padding=pad, stride=stride),
        Flatten(),
        Linear(((28 + 2 * pad - ker) // stride + 1) ** 2 * out_chan, 64),
        # Linear(28 ** 2, 64),
        Tanh(),
        Linear(64, 10),
        Softmax()
    ], CrossEntropy())
    loaders = load_minibatches()
    train(net, lr=0.001, nb_epoch=10, loaders=loaders)


if __name__ == "__main__":
    main()
