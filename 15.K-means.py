import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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


def kmeans_clustering(encoded_data, model_label, n_clusters=2, dim1=0, dim2=1, encoding_dim=4):

    os.makedirs('./result fig', exist_ok=True)

    # 取编码数据的用户选择的两个维度用于可视化
    visualization_data = encoded_data[:, [dim1, dim2]]

    # 创建图形以显示原始数据和K-means聚类结果
    plt.figure(figsize=(12, 5))

    # 绘制原始数据（用户选择的两个维度）
    plt.subplot(1, 2, 1)
    plt.scatter(visualization_data[:, 0], visualization_data[:, 1], c='blue', alpha=0.4, s=5)
    plt.title(f'Original y=0 Encoded Space (Dimensions {dim1 + 1} and {dim2 + 1})')
    plt.xlabel(f'Encoded Dimension {dim1 + 1}')
    plt.ylabel(f'Encoded Dimension {dim2 + 1}')
    plt.grid(True)

    # 在完整的encoding_dim维空间中进行K-means聚类
    kmeans_standard = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_standard_labels = kmeans_standard.fit_predict(encoded_data)

    # 可视化聚类结果（仅使用用户选择的两个维度）
    plt.subplot(1, 2, 2)
    plt.scatter(visualization_data[:, 0], visualization_data[:, 1], c=kmeans_standard_labels, alpha=0.4, s=5,
                cmap='coolwarm')

    # 如果要显示聚类中心，仅显示用户选择的两个维度
    centers = kmeans_standard.cluster_centers_
    plt.scatter(centers[:, dim1], centers[:, dim2], c='black', s=100, marker='x')

    plt.title(f'K-means Clustering (k={n_clusters}) in {encoding_dim}D Space')
    plt.xlabel(f'Encoded Dimension {dim1 + 1}')
    plt.ylabel(f'Encoded Dimension {dim2 + 1}')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        f'./result fig/kmeans_only(Model y={model_label}, dim{dim1 + 1}_dim{dim2 + 1}, k={n_clusters}, encoding_dim={encoding_dim}).png',
        dpi=300)
    plt.show()

    # 评估聚类质量
    try:
        # K-means评估
        kmeans_sil = silhouette_score(encoded_data, kmeans_standard_labels)
        kmeans_db = davies_bouldin_score(encoded_data, kmeans_standard_labels)
        kmeans_ch = calinski_harabasz_score(encoded_data, kmeans_standard_labels)

        print(f"\n==== K-means Clustering Evaluation (k={n_clusters}, {encoding_dim}D Space) ====")
        print(f"Silhouette Score: {kmeans_sil:.4f} (higher is better, range [-1,1])")
        print(f"Davies-Bouldin Score: {kmeans_db:.4f} (lower is better)")
        print(f"Calinski-Harabasz Score: {kmeans_ch:.4f} (higher is better)")

        # 计算每个簇的点数
        print("\nCluster Statistics:")
        for i in range(n_clusters):
            cluster_count = np.sum(kmeans_standard_labels == i)
            print(f"  Cluster {i}: {cluster_count} points ({cluster_count / len(kmeans_standard_labels):.2%})")

    except Exception as e:
        print(f"Error evaluating clustering quality: {e}")

    return kmeans_standard_labels


def evaluate_with_kmeans(model_path, data_path_zero, input_dim, model_label, n_clusters=2, dim1=0, dim2=1,
                         encoding_dim=4):
    # 加载模型
    model = Autoencoder(input_dim, encoding_dim=encoding_dim)
    try:
        # 尝试使用weights_only=True加载模型以避免PyTorch警告
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except:
        # 如果weights_only不支持，则使用普通加载方式
        model.load_state_dict(torch.load(model_path))
    model.eval()

    # 仅加载y=0数据
    X_zero = torch.FloatTensor(np.load(data_path_zero))

    # 获取编码结果
    with torch.no_grad():
        encoded_zero = model.encode(X_zero).numpy()

    # 应用K-means聚类
    kmeans_labels = kmeans_clustering(encoded_zero, model_label, n_clusters, dim1, dim2, encoding_dim)

    return encoded_zero, kmeans_labels


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
    dim1 = 1  # 选择第2个维度
    dim2 = 2  # 选择第3个维度

    # 确保维度在有效范围内
    if dim1 < 0 or dim1 >= encoding_dim or dim2 < 0 or dim2 >= encoding_dim:
        print(f"维度必须在0到{encoding_dim - 1}之间，将使用默认值0和1")
        dim1, dim2 = 0, 1
    elif dim1 == dim2:
        print("两个维度不能相同，将使用默认值0和1")
        dim1, dim2 = 0, 1

    # 设置聚类数量
    n_clusters = 3  # 根据评估结果设置聚类数量

    # 仅对y=0模型和数据应用K-means聚类
    print(f"\nPerforming K-means clustering on y=0 data with dimensions {dim1 + 1} and {dim2 + 1}...")
    print(f"Number of clusters: k={n_clusters}")

    encoded_zero, kmeans_labels = evaluate_with_kmeans(
        model_path=os.path.join(models_dir, f'autoencoder_zero(encoding_dim={encoding_dim}).pth'),
        data_path_zero=os.path.join(split_data_dir, 'zero_test.npy'),
        input_dim=input_dim,
        model_label=0,
        n_clusters=n_clusters,
        dim1=dim1,
        dim2=dim2,
        encoding_dim=encoding_dim
    )


if __name__ == "__main__":
    main()