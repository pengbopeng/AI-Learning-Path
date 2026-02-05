# -*- coding: utf-8 -*-
"""
【全能版】线性回归深度解析与手写实现 (Linear Regression Master File)

功能包含：
1. 从零手写 LinearRegression 类 (基于梯度下降)。
2. 场景一：单变量回归可视化 (拟合直线)。
3. 场景二：多变量回归验证 (3D平面/多维特征)。
4. 场景三：学习率(Learning Rate)过大的后果演示 (梯度爆炸)。
5. 场景四：与 Sklearn 工业界库的结果对比。

作者: PengBo (AI-Learning-Path)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error

# ==========================================
# 第一部分：手写线性回归类 (核心知识点)
# ==========================================
class MyLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化超参数
        :param learning_rate: 学习率 (Alpha) - 决定下山步子的大小
        :param n_iterations: 迭代次数 - 决定下山走多少步
        """
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []  # 记录每次迭代的损失，用来画图观察

    def fit(self, X, y):
        """
        训练模型 (Training / Fitting)
        原理：使用梯度下降法 (Gradient Descent) 不断更新权重和偏置
        """
        n_samples, n_features = X.shape
        
        # 1. 初始化参数 (通常初始化为0或小的随机数)
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        # 2. 梯度下降循环
        for i in range(self.n_iter):
            # --- 前向传播 (Forward Propagation) ---
            # 公式: y_pred = X · w + b
            y_pred = np.dot(X, self.weights) + self.bias

            # --- 计算损失 (Loss Calculation - MSE) ---
            # 公式: J = (1/m) * Σ(y_pred - y)^2
            loss = (1 / n_samples) * np.sum((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # --- 反向传播 (Backward Propagation) ---
            # 这里的数学推导是面试常考点！
            # 对 w 求导: dw = (2/m) * X.T · (y_pred - y)
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            # 对 b 求导: db = (2/m) * Σ(y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)

            # --- 参数更新 (Parameter Update) ---
            # w = w - lr * dw
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # (可选) 打印训练进度
            if i % 100 == 0:
                # print(f"Iter {i}: Loss {loss:.4f}")
                pass

    def predict(self, X):
        """预测新数据"""
        return np.dot(X, self.weights) + self.bias

# ==========================================
# 辅助函数：生成数据
# ==========================================
def generate_data(n_samples=100, noise=10):
    """生成模拟数据: y = 3x + 4 + noise"""
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1) # 特征 X (0到2之间)
    y = 4 + 3 * X + np.random.randn(n_samples, 1) * (noise/10) # 真实值 y
    return X, y

# ==========================================
# 主程序：各种场景演示
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 场景一：单变量回归 (Simple Linear Regression)")
    print("目标：拟合 y = 3x + 4")
    print("="*50)
    
    # 1. 准备数据
    X, y = generate_data()
    
    # 2. 训练我们手写的模型
    model = MyLinearRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    
    print(f"【真实参数】 w: 3, b: 4")
    print(f"【训练结果】 w: {model.weights[0][0]:.4f}, b: {model.bias:.4f}")
    
    # 3. 可视化
    plt.figure(figsize=(12, 5))
    
    # 子图1：拟合直线
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Prediction')
    plt.title('Fit Result: y = {:.2f}x + {:.2f}'.format(model.weights[0][0], model.bias))
    plt.legend()
    
    # 子图2：Loss下降曲线 (关键！面试必看)
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history)
    plt.title('Loss Curve (Training Process)')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.show()
    
    print("\n✅ 学到了什么：")
    print("1. 损失函数(Loss)随着迭代次数增加而迅速下降，最后趋于平稳（收敛）。")
    print("2. 即使有噪声干扰，梯度下降也能找到接近真实的权重和偏置。")

    # ==========================================
    
    print("\n" + "="*50)
    print("🚀 场景二：对比 Sklearn (Verify with Industry Standard)")
    print("="*50)
    
    # 1. Sklearn 训练
    sk_model = SklearnLR()
    sk_model.fit(X, y)
    
    print(f"【My Model】 w: {model.weights[0][0]:.4f}, b: {model.bias:.4f}")
    print(f"【Sklearn 】 w: {sk_model.coef_[0][0]:.4f}, b: {sk_model.intercept_[0]:.4f}")
    
    # 验证误差
    mse_my = mean_squared_error(y, model.predict(X))
    mse_sk = mean_squared_error(y, sk_model.predict(X))
    print(f"【MSE差异】: {abs(mse_my - mse_sk):.6f}")
    
    if abs(mse_my - mse_sk) < 0.1:
        print("\n✅ 结论：我们手写的算法精度达到了工业级库的水平！")
    else:
        print("\n⚠️ 结论：还需要调整学习率或迭代次数。")

    # ==========================================

    print("\n" + "="*50)
    print("🚀 场景三：多变量回归 (Multivariate Regression)")
    print("目标：拟合 y = 2*x1 + 5*x2 + 10")
    print("="*50)
    
    # 生成多维数据 (100行, 2列)
    X_multi = np.random.rand(100, 2)
    # 真实公式: y = 2*x1 + 5*x2 + 10
    w_true = np.array([[2], [5]])
    b_true = 10
    y_multi = np.dot(X_multi, w_true) + b_true + np.random.randn(100, 1) * 0.1
    
    # 训练
    model_multi = MyLinearRegression(learning_rate=0.1, n_iterations=1000)
    model_multi.fit(X_multi, y_multi)
    
    print(f"【真实权重】: [2, 5], 偏置: 10")
    print(f"【预测权重】: {model_multi.weights.flatten().round(2)}, 偏置: {model_multi.bias:.2f}")
    print("\n✅ 学到了什么：矩阵运算 (np.dot) 让我们不需要修改代码就能自动支持任意多个特征！")

    # ==========================================

    print("\n" + "="*50)
    print("💣 场景四：反面教材 - 学习率过大 (Learning Rate Explosion)")
    print("设定 learning_rate = 1.5 (步子跨太大了)")
    print("="*50)
    
    # 故意设置很大的学习率
    bad_model = MyLinearRegression(learning_rate=1.8, n_iterations=10)
    try:
        bad_model.fit(X, y)
        print("Loss history前5步:", bad_model.loss_history[:5])
    except Exception as e:
        print(f"报错了: {e}")
        
    print("\n⚠️ 观察结果：Loss 不降反升，甚至变成了 inf (无穷大) 或 nan (非数字)。")
    print("❌ 教训：梯度下降时，如果学习率太大，会跨过最低点，导致模型'发散'，永远无法收敛。")

    print("\n" + "="*50)
    print("🎉 全流程演示结束！请将此文件提交到 Git 保存。")
    print("="*50)