import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from batchgenerators.utilities.file_and_folder_operations import *

npy_path = "/opt/data/private/SFDA_ss_IF25/results/Officehome/Art_to_Clipart/CLIP_Text_Feat_Visualization-selftrain/clip_feature"

feature = np.load(join(npy_path, "text_feature.npy"))
pred = np.load(join(npy_path, "class.npy"))
domain = np.load(join(npy_path, "domain.npy"))
print(feature.shape, pred.shape, domain.shape)

# 初始化 t-SNE
tsne = TSNE(n_components=2, random_state=42)
feature_2d = tsne.fit_transform(feature)

# 设置调色板
palette = sns.color_palette("bright", n_colors=65)
color_map = {i: palette[i] for i in range(65)}

domain_unique = np.unique(domain)
markers = ['o', 's', '^', 'D']  # 4 种 marker
domain_labels = {0: 'Clipart', 1: 'Art', 2: 'RealWorld', 3: 'Product'}
marker_map = {d: markers[i] for i, d in enumerate(domain_unique)}

# 可视化 t-SNE 结果
plt.figure(figsize=(6, 6))
for i in range(65):
    for d in domain_unique:
        idx = (pred == i) & (domain == d)
        plt.scatter(feature_2d[idx, 0], feature_2d[idx, 1],
                    s=20, color=color_map[i], marker=marker_map[d],
                    facecolors='none', edgecolors=color_map[i], alpha=1.0)

# 创建图例
for d in domain_unique:
    plt.scatter([], [], color='k', marker=marker_map[d],
                label=domain_labels[d], facecolors='none', edgecolors='k')

# plt.title("t-SNE Visualization of Features by Predicted Classes")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2)
plt.axis('off')  # 隐藏坐标轴
plt.legend(loc='lower left', title="Domain")  # 显示 domain 图例
plt.tight_layout()
plt.show()

# 保存图像
plt.savefig(join(npy_path, "tsne_text.png"))