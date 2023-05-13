import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 画像1枚を32チャンネルにするから，カーネルを32変数用意してそれを裏で最適化する，ということ？
        # 28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        # conv1により25x25になる
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        # conv2により22x22になる -> これをmax_pool2d(2)にするので，floor(22/2) + 1 = 12になる
        # 64(channel) x 12 x 12 = 9216になる
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 畳み込み1
        x = F.relu(self.conv2(x))  # 畳み込み2
        x = F.max_pool2d(x, 2)  # プーリング層
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))  # 全結合層を2層にする
        output = self.fc2(x)
        return output


def main():
    learning_rate = 0.001
    batch_size = 64
    epochs = 5

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    training_data = datasets.MNIST(
        "data", train=True, download=False, transform=transform
    )
    test_data = datasets.MNIST("data", train=False, transform=transform)

    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size)

    device = torch.device("cuda")

    model = Net()
    model.to(device)

    print(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        model.train()  # これは訓練モードにするという意味だったはず
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            # 順伝播
            output = model(data)
            # 損失計算
            loss = loss_fn(output, target)
            # 誤差逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"epoch: {t+1}, step: {batch_idx+1}/{len(train_dataloader)}, loss = {loss}"
                )

        model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                correct += (
                    (output.argmax(1) == target).type(torch.float).sum().item()
                )

        print(
            f"epoch: {t+1}, test_loss = {test_loss / len(test_dataloader)}, accuracy = {correct / len(test_dataloader.dataset)}"
        )

    data = test_data[0][0]
    # plt.imshow(data.reshape((28, 28)), cmap="gray_r")

    model.eval()
    with torch.no_grad():
        # モデルには複数のデータをまとめて食わせるようになっているので，data -> [data]の形式にする．unsqueeze(0)でそれを行う
        x = data.unsqueeze(0).to(device)
        logits = model(x)
        # squeeze()するとサイズ1の次元を削除するので[data] -> dataになる
        pred = torch.softmax(logits, 1).squeeze().cpu()

    plt.bar(range(10), pred)
    plt.show()


if __name__ == "__main__":
    main()
