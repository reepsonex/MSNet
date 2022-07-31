import numpy as np
from utils.dataset import *
from torch import optim
from torchvision import transforms
from log import *
import  math
from eval import *
from Network import net
def train_net(net, device, data_path, test_path, epochs, batch_size, lr):

    train_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_dataset = ISBI_Loader(test_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=True)

    t = 50
    T = epochs

    n_t = 0.5
    lambda1 = lambda epoch: (0.9 * epoch / t + 0.001) if epoch < t else 0.01 if n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.01 else n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t)))

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()
        train_loss = 0
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(net.state_dict(), 'module.pth')
# Other codes will be uploaded after being sorted

