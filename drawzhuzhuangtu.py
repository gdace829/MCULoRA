import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({'font.size': 19})

# 数据
x = [1, 2, 3, 4, 5, 6, 7, 8]
cmu_mosi = [85.43, 85.73, 86.32, 86.40, 87.09, 87.14, 87.21, 87.31]
cmu_mosei = [78.36, 78.85, 79.36, 79.74, 79.98, 80.52, 81.23, 81.75]

# 设置柱状图宽度
width = 0.35

# 创建图形和子图
fig, ax = plt.subplots()

# 绘制 CMU-MOSI 的柱子
rects1 = ax.bar([i - width / 2 for i in x], cmu_mosi, width, label='All modalities')

# 绘制 CMU-MOSEI 的柱子
rects2 = ax.bar([i + width / 2 for i in x], cmu_mosei, width, label='Average of all modality combinations')

# 添加标题和坐标轴标签
ax.set_ylabel('Acc2', fontsize=16)
ax.set_xlabel('Rank Number', fontsize=16)
# ax.set_title('Accuracy Comparison between CMU-MOSI and CMU-MOSEI')

# 添加刻度标签
ax.set_xticks(x)
ax.set_xticklabels(x)

# 设置纵轴范围
ax.set_ylim(78, 89)
plt.subplots_adjust(bottom=0.2)
# 设置图例字体大小
legend = ax.legend(fontsize=10)

# 保存图形
plt.savefig('accuracy_comparison.png')

# 显示图形
plt.show()
    