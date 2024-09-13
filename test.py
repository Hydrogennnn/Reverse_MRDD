import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.scatter(x, y, color='blue', label='Data Points')

# 添加标题和标签
plt.title('Scatter Plot of Two Lists')
plt.xlabel('X Values')
plt.ylabel('Y Values')

# 显示图例
plt.legend()

# 存储图像为 PNG 格式
plt.savefig('scatter_plot.png')

# 显示图像
plt.show()