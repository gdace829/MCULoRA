import matplotlib.pyplot as plt

# 缺失率（横坐标）
missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# 各方法的 F1 分数（纵坐标），新增了 MCTN 和 MMIN 的数据
f1_gcnet = [85.10, 82.10, 79.90, 76.80, 74.90, 73.20, 72.40, 70.40]
f1_dicmor = [85.10, 83.50, 81.50, 79.30, 77.40, 75.80, 73.70, 72.20]
f1_imder = [85.10, 84.60, 82.40, 80.70, 78.10, 77.40, 75.50, 74.60]
f1_mcunet = [87.26, 86.90, 85.85, 84.34, 83.87, 81.20, 80.59, 79.36]
f1_mctn = [86.00, 84.50, 83.00, 81.00, 79.50, 78.00, 77.00, 76.00]  # 示例数据
f1_mmin = [86.50, 85.00, 83.50, 82.00, 80.50, 79.00, 78.00, 77.00]  # 示例数据

# 创建画布和子图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制数据
ax.plot(missing_rates, f1_mcunet, 'r-o', label='MCU-LoRA', markersize=5)
ax.plot(missing_rates, f1_gcnet, 'y-s', label='GCNet', markersize=5)
ax.plot(missing_rates, f1_dicmor, 'g-*', label='DiCMoR', markersize=5)
ax.plot(missing_rates, f1_imder, 'b-+', label='IMDer', markersize=5)
ax.plot(missing_rates, f1_mctn, 'm-d', label='MCTN', markersize=5)  # 新增MCTN
ax.plot(missing_rates, f1_mmin, 'c-^', label='MMIN', markersize=5)  # 新增MMIN

# 添加标签和标题
ax.set_xlabel('Missing Rate', fontsize=24)
ax.set_ylabel('F1 Score', fontsize=24)
ax.set_title('CMU-MOSEI', fontsize=24)
ax.tick_params(axis='both', labelsize=20)# 坐标轴

# 设置网格
ax.grid(True, linestyle='--', alpha=0.5)

# 添加图例
ax.legend(fontsize=16)

# 设置纵轴范围，使趋势更清晰
ax.set_ylim(68, 90)

# 保存图片到本地文件
plt.savefig('f1_vs_missing_rate_extendedsjs2.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()