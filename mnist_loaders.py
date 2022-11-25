import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def get_mnist_loaders(batch_size=64):
    tsfms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
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
