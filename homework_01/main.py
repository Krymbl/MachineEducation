import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

df = pd.read_csv('data.csv')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Гистограммы
df.hist(bins=50, figsize=(15, 10))
plt.tight_layout()
plt.savefig('histograms.png', dpi=150)

X_2d = X_scaled[:, [2, 6]]


def get_neighbors(X, point_idx, epsilon):
    point = X[point_idx]
    distances = np.sqrt(((X - point) ** 2).sum(axis=1))
    neighbors = np.where(distances <= epsilon)[0]
    neighbors = neighbors[neighbors != point_idx]
    return neighbors


def dbscan(X, epsilon, min_samples):
    n = X.shape[0]
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = get_neighbors(X, i, epsilon)

        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        queue = list(neighbors)

        while queue:
            j = queue.pop(0)
            if visited[j]:
                continue
            visited[j] = True
            labels[j] = cluster_id

            new_neighbors = get_neighbors(X, j, epsilon)
            if len(new_neighbors) >= min_samples:
                queue.extend(new_neighbors)

        cluster_id += 1

    return labels


print("Начали dbscan")
labels = dbscan(X_2d, epsilon=0.025, min_samples=4)
print(f'Количество кластеров: {len(set(labels)) - (1 if -1 in labels else 0)}')
print(f'Шумовых точек: {np.sum(labels == -1)}')

plt.figure(figsize=(10, 10))

noise = X_2d[labels == -1]
plt.scatter(noise[:, 1], noise[:, 0], c='black', s=5, label='Шум')

unique_labels = set(labels) - {-1}
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    points = X_2d[labels == label]
    plt.scatter(points[:, 1], points[:, 0], color=color, s=5)

plt.title(f'DBSCAN — epsilon=0.02, кластеров: {len(unique_labels)}')
plt.savefig('dbscan.png', dpi=150, bbox_inches='tight')
print('График сохранён!')

img = Image.open('dbscan.png')
img_rotated = img.rotate(-90, expand=True)
img_rotated.save('dbscan_rotated.png')
