from tqdm import tqdm

import nn
import optim
from mnist_loaders import get_mnist_loaders


def train(model, lr, nb_epoch, loaders):
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.RAdam(model.parameters(), lr=lr)
    # optimizer = optim.RMSProp(model.parameters(), lr=lr, alpha=0.9)
    # optimizer = optim.Adagrad(model.parameters(), lr=lr)

    print("Start training...")
    for epoch in range(nb_epoch):
        print(f"--- Epoch {epoch + 1}/{nb_epoch}")

        for phase in ["Train", "Test"]:
            if phase == "Train":
                model.train()
            else:
                model.eval()

            accuracy = 0
            running_loss = 0.0
            num_inputs = 0
            for (inputs, targets) in tqdm(loaders[phase]):
                # for (inputs, targets) in loaders[phase]:
                bz = inputs.shape[0]
                num_inputs += bz
                predicted = model(inputs)
                running_loss += model.loss(predicted, targets) * bz

                if phase == "Train":
                    optimizer.zero_grad()
                    model.backward()
                    optimizer.step()

                accuracy += (predicted.argmax(axis=1) == targets.argmax(axis=1)).sum()
            print(f'Phase {phase}. Loss: {running_loss / num_inputs}. Accuracy: {accuracy / num_inputs}')


def main():
    ker = 3
    pad = 1
    out_chan = 4
    stride = 1
    net = nn.Net(nn.Sequential(
        # nn.Conv2d(in_channels=1, out_channels=out_chan, kernel_size=ker, padding=pad, stride=stride),
        nn.Conv2d(in_channels=1, out_channels=out_chan * 4, kernel_size=ker, padding=pad, stride=stride),
        # nn.Conv2d(in_channels=out_chan * 4, out_channels=out_chan * 16, kernel_size=ker, padding=pad, stride=stride),
        # nn.MaxPool2d(kernel_size=ker, padding=pad, stride=stride),
        # nn.BatchNorm2d(out_chan),
        nn.Flatten(),
        nn.Linear(((28 + 2 * pad - ker) // stride + 1) ** 2 * out_chan * 4, 128),
        # nn.Linear(((28 + 2 * pad - ker) // stride + 1) ** 2, 128),
        # nn.Linear(28 ** 2, 128),
        # nn.Dropout(),
        nn.BatchNorm1d(128),
        nn.Tanh(),
        nn.Linear(128, 10),
        nn.Softmax()
    ), nn.CrossEntropyLoss())
    loaders = get_mnist_loaders(batch_size=64)
    train(net, lr=0.001, nb_epoch=10, loaders=loaders)


if __name__ == "__main__":
    main()
