import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from batchgenerators.utilities.file_and_folder_operations import *

feature_path = "/opt/data/private/SFDA_ss_IF25/results/Feat_Visualization-DIFO"

feature = np.load(join(feature_path, "features.npy"))
pred = np.load(join(feature_path, "labels.npy"))

print(feature.shape, pred.shape)

# 初始化 t-SNE
tsne = TSNE(n_components=2, random_state=42)
feature_2d = tsne.fit_transform(feature)

# 设置调色板
palette = sns.color_palette("bright", n_colors=65)
color_map = {i: palette[i] for i in range(65)}

# 可视化 t-SNE 结果
plt.figure(figsize=(6, 6))
for i in range(65):
    idx = pred == i
    plt.scatter(feature_2d[idx, 0], feature_2d[idx, 1],
                s=6, color=color_map[i], alpha=1.0)

# plt.title("t-SNE Visualization of Features by Predicted Classes")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2)
plt.axis('off')  # 隐藏坐标轴
plt.tight_layout()
plt.show()

# 保存图像
plt.savefig(join(feature_path, "tsne.png"))