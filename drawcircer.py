import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'


def plot_radar(data, methods, metrics, fontsize=20, dataset="Corel5k"):
    tab10_colors_hex = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 黄绿色
        '#17becf'  # 浅蓝色
    ]
    tab20_colors_hex = [
        '#1f77b4',  # 蓝色
        '#aad8e6',  # 浅蓝色
        '#ff7f0e',  # 橙色
        '#ffbb77',  # 浅橙色
        '#2ca02c',  # 绿色
        '#96e8ac',  # 浅绿色
        '#d62728',  # 红色
        '#ff9994',  # 浅红色
        '#9467bd',  # 紫色
        '#c4a6d4',  # 浅紫色
        '#8c564b',  # 棕色
        '#c3a293',  # 浅棕色
        '#e377c2',  # 粉色
        '#f5b5d2',  # 浅粉色
        '#7f7f7f',  # 灰色
        '#c6c6c6',  # 浅灰色
        '#bcbd22',  # 黄绿色
        '#d9db8c',  # 浅黄绿色
        '#17becf',  # 浅蓝色
        '#9ec8e4'  # 浅蓝绿色
    ]

    # 对每个指标的数据进行归一化，确保最小值在圆心，最大值在最外层
    data_normalized = np.zeros_like(data)
    for i in range(len(metrics)):
        data_normalized[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))


    # 设置雷达图的角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()

    # 将数据循环连接到第一个点，使其形成闭环
    data_normalized = np.concatenate((data_normalized, data_normalized[[0], :]), axis=0)
    angles += angles[:1]

    # 创建一个雷达图
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])  # 设置径向刻度
    ax.set_rlabel_position(0)  # 设置标签位置
    ax.grid(True, linestyle='--', linewidth=1, color='#808080', alpha=0.7)  # 添加网格线
    # ax.spines['polar'].set_visible(False)  # 隐藏极坐标的边框

    # 为每个方法绘制雷达图
    for i, method in enumerate(methods):
        ax.plot(angles, data_normalized[:, i], linewidth=3, label=method, marker='o', linestyle='-', markersize=10, color=tab10_colors_hex[8-i])
        ax.fill(angles, data_normalized[:, i], color=tab10_colors_hex[8-i], alpha=0.08)

    # 设置雷达图的标签
    ax.set_yticklabels([])

    label_with_range = []
    for i in range(len(metrics)):
        metric_min = np.min(data[i])
        metric_max = np.max(data[i])
        label_with_range.append(f"{metrics[i]}\n[{metric_min:.2f},{metric_max:.2f}]")

    # plt.tight_layout()
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    # ax.set_xticklabels(label_with_range, fontsize=fontsize)
    # ax.tick_params(axis='x', pad=39)  # 调整 pad 的值以改变距离
    pads = [1.36, 1.32, 1.32, 1.36, 1.23, 1.25]
    for i, label in enumerate(label_with_range):
        angle = angles[i]
        # 在指定位置添加标签，调整 x、y 坐标
        ax.text(angle, pads[i], label, horizontalalignment='center', verticalalignment='center', fontsize=fontsize+7,)


    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.2), fontsize=fontsize-5, )

    plt.savefig(f"./radar_{dataset}.jpg", bbox_inches='tight')
    # plt.savefig(f"./radar_{dataset}.pdf", bbox_inches='tight')
    # 显示雷达图
    # plt.show()
    plt.close()


if __name__ == '__main__':
    methods = ['MCULora(our)','EURA', 'MoMKE', 'DiCMoR', 'IMDer','GCNet']
    metrics = ['A', 'T', 'V', 'A + V', 'A + T', 'T + V']
    data = np.array([
        [0.6818, 0.6454, 0.6592, 0.6294,0.6380,0.6020],  # Precision
        [0.8635, 0.8534, 0.8584, 0.8426,0.8450,0.8300],  # Recall
        [0.6735, 0.6635, 0.6491, 0.6365,0.6390,0.6190],  # F1-Score
        [0.7197, 0.6653, 0.6586, 0.6524,0.6490,0.6410],  # Accuracy
        [0.8660, 0.8562, 0.8603, 0.8504,0.8510,0.8430],        # Time (lower is better)
        [0.8712, 0.8602, 0.8442, 0.8495,0.8500,0.8430]       # Complexity (lower is better)
    ])
    plot_radar(data, methods, metrics, fontsize=25, dataset="dataset")
