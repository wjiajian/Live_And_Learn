# 利用梯度下降算法，实现一个多元线性回归
'''
x1: 温度 ;x2: 价格 ;y: 销量
w0: 截距 ;w1:温度权重 ;w2: 价格权重
预测销量 y_pred = w0 + w1*x1 + w2*x2
损失函数 loss = sum((y_pred[j] - y[j]) ** 2 for j in range(len(y))) / len(y)
'''
# Feature 数据
X = [[10, 3], [20, 3], [25, 3], [28, 2.5], [30, 2], [35, 2.5], [40, 2.5]]
y = [60, 85, 100, 120, 140, 145, 163]  # Label 数据
# 初始化参数
w = [0.0, 0.0, 0.0]  # w0, w1, w2
lr = 0.0001  # 学习率，学习率太大的话，训练过程不会收敛， loss 值可能会越来越大，直到程序出错。
num_iterations = 10000  # 迭代次数

for i in range(num_iterations):
    # 预测值
    y_pred = [w[0]+w[1]*x[0]+w[2]*x[1] for x in X ]

    # 计算损失
    loss = sum((y_pred[j] - y[j]) ** 2 for j in range(len(y))) / len(y)

    # 计算梯度
    grad_w0 = 2 * sum(y_pred[j] - y[j] for j in range(len(y))) / len(y)
    grad_w1 = 2 * sum((y_pred[j] - y[j]) * X[j][0] for j in range(len(y))) / len(y)
    grad_w2 = 2 * sum((y_pred[j] - y[j]) * X[j][1] for j in range(len(y))) / len(y)
    
    # 更新参数
    w[0] -= lr * grad_w0
    w[1] -= lr * grad_w1
    w[2] -= lr * grad_w2

    # 打印损失
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss}")

# 输出最终参数
print(f"Final parameters: w0 = {w[0]}, w1 = {w[1]}, w2 = {w[2]}")