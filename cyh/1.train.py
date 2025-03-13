import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=2):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def split_and_save_data(data_path, save_dir, class_name):
    """
    将数据集分割并保存为训练集、验证集和测试集
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    X = np.load(data_path)
    X = torch.FloatTensor(X)
    dataset = TensorDataset(X, X)

    # 计算分割大小
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # 随机分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 提取数据
    def extract_data(dataset):
        all_data = []
        for data, _ in dataset:
            all_data.append(data.numpy())
        return np.array(all_data)

    train_data = extract_data(train_dataset)
    val_data = extract_data(val_dataset)
    test_data = extract_data(test_dataset)

    # 保存分割后的数据集
    np.save(os.path.join(save_dir, f'{class_name}_train.npy'), train_data)
    np.save(os.path.join(save_dir, f'{class_name}_val.npy'), val_data)
    np.save(os.path.join(save_dir, f'{class_name}_test.npy'), test_data)

    print(f"{class_name} 数据集分割完成:")
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")

    return train_data.shape[1]  # 返回输入维度


def create_data_loader(data_path, batch_size):

    X = np.load(data_path)
    X = torch.FloatTensor(X)
    dataset = TensorDataset(X, X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate_model(model, dataloader, criterion):

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_X, _ in dataloader:
            output = model(batch_X)
            loss = criterion(output, batch_X)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_autoencoder(train_data_path, val_data_path, model_save_path, input_dim,
                      epochs=200, batch_size=256, learning_rate=0.0001):

    # 创建数据加载器
    train_loader = create_data_loader(train_data_path, batch_size)
    val_loader = create_data_loader(val_data_path, batch_size)

    # 初始化模型
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 用于早停的变量
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # 记录训练过程
    train_losses = []
    val_losses = []

    print(f"\n开始训练...")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, _ in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        val_loss = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}')

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    print(f"模型已保存到: {model_save_path}")

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def main():
    # 设置路径
    data_dir = '../data'
    split_data_dir = './split_data'
    models_dir = './models'
    os.makedirs(models_dir, exist_ok=True)

    # 处理 y=0 的数据
    print("\n处理 y=0 的数据...")
    input_dim_zero = split_and_save_data(
        data_path=os.path.join(data_dir, 'X_zero.npy'),
        save_dir=split_data_dir,
        class_name='zero'
    )

    # 训练 y=0 的模型
    train_autoencoder(
        train_data_path=os.path.join(split_data_dir, 'zero_train.npy'),
        val_data_path=os.path.join(split_data_dir, 'zero_val.npy'),
        model_save_path=os.path.join(models_dir, 'autoencoder_zero(encoding_dim=2).pth'),
        input_dim=input_dim_zero
    )

    # 处理 y=1 的数据
    print("\n处理 y=1 的数据...")
    input_dim_one = split_and_save_data(
        data_path=os.path.join(data_dir, 'X_one.npy'),
        save_dir=split_data_dir,
        class_name='one'
    )

    # 训练 y=1 的模型
    train_autoencoder(
        train_data_path=os.path.join(split_data_dir, 'one_train.npy'),
        val_data_path=os.path.join(split_data_dir, 'one_val.npy'),
        model_save_path=os.path.join(models_dir, 'autoencoder_one(encoding_dim=2)(epoch=200).pth'),
        input_dim=input_dim_one,
        #epochs=1000
    )


if __name__ == "__main__":
    main()