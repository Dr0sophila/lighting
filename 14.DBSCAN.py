import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=4):
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

    def encode(self, x):
        return self.encoder(x)


def dbscan_clustering(encoded_data, model_label, dim1=0, dim2=1, encoding_dim=4, eps=0.65, min_samples=500):
    """
    执行DBSCAN聚类并显示原始数据和聚类结果

    参数:
        encoded_data: 编码后的数据
        model_label: 模型标签
        dim1: 第一个要可视化的维度 (默认为0)
        dim2: 第二个要可视化的维度 (默认为1)
        encoding_dim: 编码的总维度
        eps: DBSCAN的eps参数（邻域半径）
        min_samples: DBSCAN的min_samples参数（核心点的最小样本数）
    """
    os.makedirs('./result fig', exist_ok=True)

    # 取编码数据的选择的两个维度用于可视化
    visualization_data = encoded_data[:, [dim1, dim2]]

    # 创建图形以显示原始数据和DBSCAN聚类结果
    plt.figure(figsize=(12, 5))

    # 绘制原始数据（选择的两个维度）
    plt.subplot(1, 2, 1)
    plt.scatter(visualization_data[:, 0], visualization_data[:, 1], c='blue', alpha=0.4, s=5)
    plt.title(f'Original y=0 Encoded Space (Dimensions {dim1 + 1} and {dim2 + 1})')
    plt.xlabel(f'Encoded Dimension {dim1 + 1}')
    plt.ylabel(f'Encoded Dimension {dim2 + 1}')
    plt.grid(True)

    # 在完整的encoding_dim维空间中进行DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(encoded_data)

    # 计算聚类数量（不包括噪声点，噪声点标签为-1）
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    # 可视化聚类结果（仅使用选择的两个维度）
    plt.subplot(1, 2, 2)
    plt.scatter(visualization_data[:, 0], visualization_data[:, 1], c=dbscan_labels, alpha=0.4, s=5,
                cmap='coolwarm')

    plt.title(f'DBSCAN Clustering (clusters={n_clusters}) in {encoding_dim}D Space')
    plt.xlabel(f'Encoded Dimension {dim1 + 1}')
    plt.ylabel(f'Encoded Dimension {dim2 + 1}')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        f'./result fig/dbscan(Model y={model_label}, dim{dim1 + 1}_dim{dim2 + 1}, eps={eps}, ms={min_samples}).png',
        dpi=300)
    plt.show()

    # 评估聚类质量（如果有超过1个聚类）
    try:
        # 计算噪声点的数量和比例
        noise_count = np.sum(dbscan_labels == -1)
        print("\nDBSCAN Clustering Results:")
        print(f"  Number of clusters found: {n_clusters}")
        print(f"  Noise points: {noise_count} points ({noise_count / len(dbscan_labels):.2%})")

        # 计算每个簇的点数
        unique_labels = set(dbscan_labels)
        for label in sorted(unique_labels):
            if label == -1:
                continue  # 已经报告了噪声点
            count = np.sum(dbscan_labels == label)
            print(f"  Cluster {label}: {count} points ({count / len(dbscan_labels):.2%})")

        # 如果有超过1个聚类，并且没有所有点都是噪声点，尝试计算评估指标
        if n_clusters > 1 and noise_count < len(dbscan_labels):
            # 创建仅包含非噪声点的数据子集用于评估
            non_noise_mask = dbscan_labels != -1
            if np.sum(non_noise_mask) > 1:  # 确保有足够的非噪声点
                non_noise_data = encoded_data[non_noise_mask]
                non_noise_labels = dbscan_labels[non_noise_mask]

                # 检查是否至少有两个不同的标签用于计算
                if len(set(non_noise_labels)) > 1:
                    dbscan_sil = silhouette_score(non_noise_data, non_noise_labels)
                    dbscan_db = davies_bouldin_score(non_noise_data, non_noise_labels)
                    dbscan_ch = calinski_harabasz_score(non_noise_data, non_noise_labels)

                    print(f"\n==== DBSCAN Clustering Evaluation ({encoding_dim}D Space, Non-noise Points) ====")
                    print(f"Silhouette Score: {dbscan_sil:.4f} (higher is better, range [-1,1])")
                    print(f"Davies-Bouldin Score: {dbscan_db:.4f} (lower is better)")
                    print(f"Calinski-Harabasz Score: {dbscan_ch:.4f} (higher is better)")

    except Exception as e:
        print(f"Error evaluating clustering quality: {e}")

    return dbscan_labels


def evaluate_with_dbscan(model_path, data_path_zero, input_dim, model_label, dim1=0, dim2=1, encoding_dim=4, eps=0.5,
                         min_samples=50):
    """
    使用DBSCAN评估数据

    参数:
        model_path: 模型路径
        data_path_zero: 数据路径
        input_dim: 输入维度
        model_label: 模型标签
        dim1: 第一个要可视化的维度 (默认为0)
        dim2: 第二个要可视化的维度 (默认为1)
        encoding_dim: 编码的总维度 (默认为4)
        eps: DBSCAN的eps参数（邻域半径）
        min_samples: DBSCAN的min_samples参数（核心点的最小样本数）
    """
    # 加载模型
    model = Autoencoder(input_dim, encoding_dim=encoding_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 仅加载y=0数据
    X_zero = torch.FloatTensor(np.load(data_path_zero))

    # 获取编码结果
    with torch.no_grad():
        encoded_zero = model.encode(X_zero).numpy()

    # 应用DBSCAN聚类
    dbscan_labels = dbscan_clustering(encoded_zero, model_label, dim1, dim2, encoding_dim, eps, min_samples)

    return encoded_zero, dbscan_labels


def main():
    # 设置路径
    split_data_dir = './split_data'
    models_dir = './models'

    # 获取数据维度
    X_zero = np.load(os.path.join(split_data_dir, 'zero_test.npy'))
    input_dim = X_zero.shape[1]

    # 设置编码维度
    encoding_dim = 4

    # 直接在代码中设置要可视化的维度
    dim1 = 1  # 选择第3个维度
    dim2 = 2  # 选择第4个维度

    # 确保维度在有效范围内
    if dim1 < 0 or dim1 >= encoding_dim or dim2 < 0 or dim2 >= encoding_dim:
        print(f"维度必须在0到{encoding_dim - 1}之间，将使用默认值0和1")
        dim1, dim2 = 0, 1
    elif dim1 == dim2:
        print("两个维度不能相同，将使用默认值0和1")
        dim1, dim2 = 0, 1

    # 设置DBSCAN参数
    eps = 0.5  # 邻域半径
    min_samples = 60  # 核心点的最小样本数

    # 仅对y=0模型和数据应用DBSCAN聚类
    print(f"\nPerforming DBSCAN clustering on y=0 data with dimensions {dim1 + 1} and {dim2 + 1}...")
    print(f"Parameters: eps={eps}, min_samples={min_samples}")

    encoded_zero, dbscan_labels = evaluate_with_dbscan(
        model_path=os.path.join(models_dir, f'autoencoder_zero(encoding_dim={encoding_dim}).pth'),
        data_path_zero=os.path.join(split_data_dir, 'zero_test.npy'),
        input_dim=input_dim,
        model_label=0,
        dim1=dim1,
        dim2=dim2,
        encoding_dim=encoding_dim,
        eps=eps,
        min_samples=min_samples
    )


if __name__ == "__main__":
    main()