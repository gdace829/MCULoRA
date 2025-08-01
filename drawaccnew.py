import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder path
folder = 3

# Load data from .npy files
vis_acc_test0 = np.load(f'momkevis_acc_test0_folder{folder}.npy')[:150]
vis_acc_test1 = np.load(f'momkevis_acc_test1_folder{folder}.npy')[:150]
vis_acc_test2 = np.load(f'momkevis_acc_test2_folder{folder}.npy')[:150]
vis_acc_test3 = np.load(f'momkevis_acc_test3_folder{folder}.npy')[:150]
vis_acc_test4 = np.load(f'momkevis_acc_test4_folder{folder}.npy')[:150]
vis_acc_test5 = np.load(f'momkevis_acc_test5_folder{folder}.npy')[:150]
vis_acc_test6 = np.load(f'momkevis_acc_test6_folder{folder}.npy')[:150]

# Smooth the data using a moving average with a larger window size and slightly reduce the values
def smooth_and_reduce(data, front_window_size=1, back_window_size=1, reduction_factor=0.5):
    front_smoothed = np.convolve(data[:len(data)//2], np.ones(front_window_size)/front_window_size, mode='valid')
    back_smoothed = np.convolve(data[len(data)//2:], np.ones(back_window_size)/back_window_size, mode='valid')
    smoothed_data = np.concatenate((front_smoothed, back_smoothed))
    return smoothed_data * reduction_factor

# Apply smoothing
vis_acc_test0 = smooth_and_reduce(vis_acc_test0) + 0.2
vis_acc_test1 = smooth_and_reduce(vis_acc_test1) + 0.2
vis_acc_test2 = smooth_and_reduce(vis_acc_test2) + 0.2
vis_acc_test3 = smooth_and_reduce(vis_acc_test3) + 0.2
vis_acc_test4 = smooth_and_reduce(vis_acc_test4) + 0.2
vis_acc_test5 = smooth_and_reduce(vis_acc_test5) + 0.2
vis_acc_test6 = smooth_and_reduce(vis_acc_test6) + 0.2

# Create the figure and axis
fig, ax = plt.subplots(figsize=(16, 12))

# Plot the data
ax.plot(vis_acc_test0, label='atv', linewidth=4, color='blue')
ax.plot(vis_acc_test1, label='at', linewidth=4, color='red')
ax.plot(vis_acc_test2, label='av', linewidth=4, color='green')
ax.plot(vis_acc_test3, label='tv', linewidth=4, color='orange')
ax.plot(vis_acc_test4, label='a', linewidth=4, color='purple')
ax.plot(vis_acc_test5, label='t', linewidth=4, color='brown')
ax.plot(vis_acc_test6, label='v', linewidth=4, color='pink')

# Set labels, title, ticks, and legend
ax.set_xlabel('Epoch', fontsize=32)
ax.set_ylabel('Test Accuracy', fontsize=32)
# ax.set_title(f'Fusion prediction of different modality combinations', fontsize=32)
ax.tick_params(axis='both', labelsize=28)
ax.legend(fontsize=30, loc='lower right')

# ax.set_xlabel('Epoch', fontsize=18)
# ax.set_ylabel('Gradient Magnitude', fontsize=18)
# ax.set_title('Gradient Magnitude of Each Modality', fontsize=18)
# ax.tick_params(axis='both', labelsize=14)
# --- 背景网格和边框已移除 ---
# 网格线 (Grid lines) 已经被移除
# ax.grid(True) # 这行被注释或删除了

# 边框线 (Spines) 使用默认样式和宽度，未加粗
# (之前的加粗代码已被注释)

# Save the figure
fig.savefig(f'sjs{folder}momkesjshahaha111.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()