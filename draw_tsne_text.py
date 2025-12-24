import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import colorsys
from batchgenerators.utilities.file_and_folder_operations import *

class Color_Generator:
    @staticmethod
    def generate_distinct_colors(n):
        """调色板方案1：生成n个区分度高的颜色"""
        colors = []
        for i in range(n):
            # 在HSV空间均匀分布色相
            hue = i / n

            # 高饱和度设置：0.85-1.0
            # 在基础高饱和上加入轻微变化，增加区分度
            base_saturation = 0.95
            variation = 0.05 * (i % 3) / 3  # 微小变化
            saturation = min(1.0, base_saturation + variation)
            
            # 亮度：0.8-0.95（避免过亮发白）
            base_value = 0.88
            value_variation = 0.07 * ((i // 2) % 3) / 3
            value = min(1.0, base_value + value_variation)

            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors
    
    @staticmethod
    def generate_high_contrast_colors(n_classes, 
                                 sat_range=(0.4, 1.0),  # 扩大饱和度范围
                                 val_range=(0.4, 0.95)): # 扩大亮度范围
        """
        调色板方案2：生成高对比度的颜色，饱和度和亮度剧烈变化
        
        参数:
        n_classes: 类别数量
        sat_range: 饱和度范围 (min, max)
        val_range: 亮度范围 (min, max)
        """
        colors = []
        sat_min, sat_max = sat_range
        val_min, val_max = val_range
        
        # 使用质数间隔避免周期性重复
        hue_step = 0.6180339887  # 黄金比例倒数
        sat_step = 0.3
        val_step = 0.4
        
        for i in range(n_classes):
            # 1. 色相：使用黄金比例分布
            hue = (i * hue_step) % 1.0
            
            # 2. 饱和度：剧烈变化模式
            # 使用sin函数创建多个波动周期
            sat_phase = i * sat_step
            sat_variation = np.sin(sat_phase) * 0.25 + 0.5  # -0.25到0.25变化
            saturation = sat_min + (sat_max - sat_min) * (0.5 + sat_variation * 0.5)
            saturation = np.clip(saturation, sat_min, sat_max)
            
            # 3. 亮度：不同的剧烈变化模式
            val_phase = i * val_step
            val_variation = np.cos(val_phase) * 0.3 + 0.5  # -0.3到0.3变化
            value = val_min + (val_max - val_min) * (0.5 + val_variation * 0.5)
            value = np.clip(value, val_min, val_max)
            
            # 4. 转换为RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return colors
    
    @staticmethod
    def generate_grouped_contrast_colors(n_classes):
        """
        调色板方案3：分组生成高对比度颜色
        """
        colors = []
        
        # 定义多个分组，每组使用不同的饱和度和亮度模式
        group_params = [
            {'sat_range': (0.3, 0.6), 'val_range': (0.7, 0.95)},  # 低饱和，高亮
            {'sat_range': (0.8, 1.0), 'val_range': (0.3, 0.6)},  # 高饱和，低亮
            {'sat_range': (0.6, 0.9), 'val_range': (0.8, 1.0)},  # 中高饱和，高亮
            {'sat_range': (0.4, 0.7), 'val_range': (0.4, 0.7)},  # 中饱和，中亮
            {'sat_range': (0.9, 1.0), 'val_range': (0.9, 1.0)},  # 最高饱和，最高亮
        ]
        
        for i in range(n_classes):
            # 每个颜色选择不同的分组模式
            group_idx = i % len(group_params)
            hue_idx = i // len(group_params)
            
            # 色相：确保每个分组内的颜色也不同
            hue = (hue_idx * 0.1618) % 1.0  # 使用黄金比例相关值
            
            # 从对应分组获取饱和度和亮度范围
            params = group_params[group_idx]
            sat_range = params['sat_range']
            val_range = params['val_range']
            
            # 在组内创建变化
            group_size = n_classes // len(group_params) + 1
            position_in_group = hue_idx % group_size
            
            # 饱和度：在组内剧烈变化
            saturation = sat_range[0] + (sat_range[1] - sat_range[0]) * \
                        ((position_in_group * 0.3) % 1.0)
            
            # 亮度：与饱和度反向变化以增加对比
            value = val_range[0] + (val_range[1] - val_range[0]) * \
                ((1.0 - (position_in_group * 0.2) % 1.0))
            
            # 添加随机微扰
            saturation = np.clip(saturation + np.random.uniform(-0.05, 0.05), 0, 1)
            value = np.clip(value + np.random.uniform(-0.05, 0.05), 0, 1)
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return colors
    
    @staticmethod
    def generate_multi_strategy_colors(n_classes):
        """
        调色板方案4：根据色相分区应用不同的饱和度和亮度策略
        """
        colors = []
        
        for i in range(n_classes):
            hue = i / n_classes
            
            # 根据色相区域选择不同的策略
            if hue < 0.166:  # 红色区域 (0-60度)
                # 策略1: 保持高饱和度，亮度大幅变化
                saturation = 0.9 + 0.1 * (i % 3) / 3
                value = 0.3 + 0.7 * ((i * 7) % 11) / 11
                
            elif hue < 0.333:  # 黄绿区域 (60-120度)
                # 策略2: 饱和度和亮度都剧烈变化
                saturation = 0.4 + 0.6 * ((i * 5) % 8) / 8
                value = 0.5 + 0.5 * ((i * 3) % 7) / 7
                
            elif hue < 0.5:  # 青色区域 (120-180度)
                # 策略3: 中等饱和度，超高对比亮度
                saturation = 0.7 + 0.3 * ((i * 2) % 5) / 5
                value = 0.2 + 0.8 * ((i % 4) / 4)
                
            elif hue < 0.666:  # 蓝色区域 (180-240度)
                # 策略4: 从低到高的饱和度，中等亮度变化
                saturation = 0.3 + 0.7 * (i % 6) / 6
                value = 0.6 + 0.4 * ((i * 4) % 9) / 9
                
            elif hue < 0.833:  # 紫色区域 (240-300度)
                # 策略5: 高饱和度，亮度在低和中之间切换
                saturation = 0.85 + 0.15 * ((i % 5) / 5)
                value = 0.4 if (i % 3 == 0) else 0.9
                
            else:  # 品红区域 (300-360度)
                # 策略6: 饱和度和亮度都跳跃变化
                saturation = 0.5 if (i % 4 == 0) else 1.0
                value = 0.3 if (i % 5 < 2) else 0.8
            
            # 确保边界
            saturation = np.clip(saturation, 0.2, 1.0)
            value = np.clip(value, 0.2, 1.0)
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return colors
    
    @staticmethod
    def generate_noisy_contrast_colors(n_classes, seed=42):
        """
        调色板方案5：使用Perlin噪声生成复杂变化的颜色
        """
        np.random.seed(seed)
        colors = []
        
        # 生成随机但可重复的变化序列
        sat_noise = np.random.uniform(-0.4, 0.4, n_classes)
        val_noise = np.random.uniform(-0.4, 0.4, n_classes)
        
        # 添加二次变化
        sat_noise2 = np.sin(np.linspace(0, 8*np.pi, n_classes)) * 0.3
        val_noise2 = np.cos(np.linspace(0, 6*np.pi, n_classes)) * 0.3
        
        for i in range(n_classes):
            # 基础色相
            hue = (i * 0.6180339887) % 1.0
            
            # 剧烈变化的饱和度：基础值+噪声
            base_sat = 0.5 + 0.3 * np.sin(i * 0.5)
            saturation = base_sat + sat_noise[i] + sat_noise2[i]
            saturation = np.clip(saturation, 0.2, 1.0)
            
            # 剧烈变化的亮度：与饱和度负相关增加对比
            base_val = 0.6 + 0.2 * np.cos(i * 0.7)
            value = base_val + val_noise[i] + val_noise2[i]
            # 尝试与饱和度负相关
            if i % 3 == 0:
                value = 1.0 - saturation * 0.7
            value = np.clip(value, 0.2, 1.0)
            
            # 特别处理：每10个颜色创建一个极端对比
            if i % 10 == 0:
                saturation = 1.0 if saturation > 0.5 else 0.2
                value = 0.2 if value > 0.5 else 1.0
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return colors


if __name__ == "__main__":
    npy_path = "/opt/data/private/SFDA_ss_IF25/results/Officehome/Clipart_to_Art/CLIP_Text_Feat_Visualization-oracle/clip_feature"

    feature = np.load(join(npy_path, "text_feature.npy"))
    pred = np.load(join(npy_path, "class.npy"))
    domain = np.load(join(npy_path, "domain.npy"))
    print(feature.shape, pred.shape, domain.shape)

    # 初始化 t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    feature_2d = tsne.fit_transform(feature)

    # 设置调色板
    palette = Color_Generator.generate_grouped_contrast_colors(65)
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