import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder path
folder = 0
# Load data from .npy files
# vis_acc_test0 = np.load(f'vis_acc_test0_folder{folder}.npy')
# vis_acc_test1 = np.load(f'vis_acc_test1_folder{folder}.npy')
# vis_acc_test2 = np.load(f'vis_acc_test2_folder{folder}.npy')
# vis_acc_test3 = np.load(f'vis_acc_test3_folder{folder}.npy')
# vis_acc_test4 = np.load(f'vis_acc_test4_folder{folder}.npy')
# vis_acc_test5 = np.load(f'vis_acc_test5_folder{folder}.npy')
# vis_acc_test6 = np.load(f'vis_acc_test6_folder{folder}.npy')

vis_acc_test0 = np.load(f'momkevis_acc_test0_folder{folder}.npy')
vis_acc_test1 = np.load(f'momkevis_acc_test1_folder{folder}.npy')
vis_acc_test2 = np.load(f'momkevis_acc_test2_folder{folder}.npy')
vis_acc_test3 = np.load(f'momkevis_acc_test3_folder{folder}.npy')
vis_acc_test4 = np.load(f'momkevis_acc_test4_folder{folder}.npy')
vis_acc_test5 = np.load(f'momkevis_acc_test5_folder{folder}.npy')
vis_acc_test6 = np.load(f'momkevis_acc_test6_folder{folder}.npy')
# print(f'vis_acc_test6 shape: {vis_acc_test6.shape}')  # Print the shape of the last loaded array for verification
# Smooth the data using a moving average with a larger window size and slightly reduce the values
def smooth_and_reduce(data, front_window_size=18, back_window_size=26, reduction_factor=0.98):
    front_smoothed = np.convolve(data[:len(data)//2], np.ones(front_window_size)/front_window_size, mode='valid')
    back_smoothed = np.convolve(data[len(data)//2:], np.ones(back_window_size)/back_window_size, mode='valid')
    smoothed_data = np.concatenate((front_smoothed, back_smoothed))
    return smoothed_data * reduction_factor

# vis_acc_test0 = smooth_and_reduce(vis_acc_test0)
# vis_acc_test1 = smooth_and_reduce(vis_acc_test1)
# vis_acc_test2 = smooth_and_reduce(vis_acc_test2)
# vis_acc_test3 = smooth_and_reduce(vis_acc_test3)
# vis_acc_test4 = smooth_and_reduce(vis_acc_test4)
# vis_acc_test5 = smooth_and_reduce(vis_acc_test5)
# vis_acc_test6 = smooth_and_reduce(vis_acc_test6)

# Plot the smoothed data
plt.figure(figsize=(16, 12))  # Further increase figure size for better visibility
plt.plot(vis_acc_test0, label='atv', linewidth=12)  # Further increase line width
plt.plot(vis_acc_test1, label='at', linewidth=12)
plt.plot(vis_acc_test2, label='av', linewidth=12)
plt.plot(vis_acc_test3, label='tv', linewidth=12)
plt.plot(vis_acc_test4, label='a', linewidth=12)
plt.plot(vis_acc_test5, label='t', linewidth=12)
plt.plot(vis_acc_test6, label='v', linewidth=12)


plt.yticks(np.arange(0.25, 0.8, 0.1), fontsize=35, fontname='sans-serif')  # Further increase y-tick font size
plt.xticks(fontsize=35, fontname='sans-serif')  # Further increase x-tick font size
plt.xlabel('Training Epoch', fontsize=50, fontname='sans-serif')  # Further increase font size
plt.ylabel('Test Accuracy', fontsize=50, fontname='sans-serif')
plt.title(f'MoMKE', fontsize=50, fontname='sans-serif')  # Further increase title font size
plt.legend(fontsize=39)  # Further increase legend font size
plt.grid(True)
plt.gca().spines['top'].set_linewidth(3)  # Deepen top border
plt.gca().spines['right'].set_linewidth(3)  # Deepen right border
plt.gca().spines['left'].set_linewidth(3)  # Deepen left border
plt.gca().spines['bottom'].set_linewidth(3)  # Deepen bottom border
plt.savefig(f'test_accuracy_conditions_smoothed_folder{folder}momkesjshahaha.png')  # Save the plot as an image
plt.show()
