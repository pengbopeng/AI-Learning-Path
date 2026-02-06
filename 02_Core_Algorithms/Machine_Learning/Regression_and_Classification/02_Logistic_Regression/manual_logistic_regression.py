# -*- coding: utf-8 -*-
"""
【手撕逻辑回归】从Sigmoid到二分类实战 (Logistic Regression from Scratch)

核心目标：
1. 可视化 Sigmoid 函数 (AI界的"压缩机")。
2. 实现逻辑回归算法，解决二分类问题。
3. 场景演示：根据"学习时长"预测"考试是否及格"。

作者: PengBo (AI-Learning-Path)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.metrics import accuracy_score

# ==========================================
# 核心组件：Sigmoid 函数
# ==========================================
def sigmoid(z):
    """
    逻辑回归的灵魂：将任意实数压缩到 (0, 1) 区间
    公式: g(z) = 1 / (1 + e^-z)
    """
    return 1 / (1 + np.exp(-z))

# ==========================================
# 第一部分：手写逻辑回归类
# ==========================================
class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降
        for i in range(self.n_iter):
            # 1. 线性部分: z = w*x + b
            linear_model = np.dot(X, self.weights) + self.bias
            
            # 2. 激活部分 (压缩成概率): y_pred = sigmoid(z)
            y_pred = sigmoid(linear_model)

            # 3. 计算损失 (Log Loss / 交叉熵损失) - 面试必问！
            # 也就是：预测越准，Loss越小；预测越离谱，Loss无穷大
            # 为了防止 log(0) 报错，通常加一个极小值 epsilon
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = - (1/n_samples) * np.sum(y * np.log(y_pred_clipped) + (1-y) * np.log(1-y_pred_clipped))
            self.loss_history.append(loss)

            # 4. 反向传播 (求导)
            # 惊人的巧合：这里的导数公式和线性回归一模一样！
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 5. 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        """输出概率值 (0.2, 0.8, 0.9...)"""
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X):
        """输出类别 (0 或 1)"""
        y_pred_proba = self.predict_proba(X)
        # 概率 > 0.5 判为 1 (通过)，否则判为 0 (挂科)
        return [1 if i > 0.5 else 0 for i in y_pred_proba]

# ==========================================
# 主程序：场景演示
# ==========================================
if __name__ == "__main__":
    
    # --- 1. 理解 Sigmoid ---
    print("正在绘制 Sigmoid 函数...")
    z = np.linspace(-10, 10, 100)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(z, sigmoid(z), color='blue', linewidth=3)
    plt.axvline(x=0, color='k', linestyle='--') # 中线
    plt.axhline(y=0.5, color='r', linestyle='--') # 0.5 阈值线
    plt.title("Sigmoid Function (The Activation)")
    plt.grid(True)
    
    # --- 2. 构造数据：学习时间 vs 考试结果 ---
    print("构造考试数据...")
    # 学习时间 (特征)
    X = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.75, 2.0, 2.25, 2.5, 
                  2.75, 3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5]).reshape(-1, 1)
    # 是否通过 (标签 0=挂科, 1=通过)
    # 假设学习时间超过 2.5小时 大概率能过
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 
                  1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

    # --- 3. 训练模型 ---
    model = MyLogisticRegression(learning_rate=0.1, n_iterations=3000)
    model.fit(X, y)
    
    # --- 4. 预测与评估 ---
    # 假如有个学生学了 4 小时
    test_student = np.array([[4.0]])
    prob = model.predict_proba(test_student)[0]
    print(f"\n【预测】学 4 小时，及格概率是: {prob:.4f} -> {'Pass' if prob>0.5 else 'Fail'}")
    
    # 对比 Sklearn
    sk_model = SklearnLogReg()
    sk_model.fit(X, y)
    sk_acc = accuracy_score(y, sk_model.predict(X))
    my_acc = accuracy_score(y, model.predict(X))
    print(f"\n【准确率对比】 手写版: {my_acc*100}% | Sklearn: {sk_acc*100}%")

    # --- 5. 可视化分类边界 ---
    plt.subplot(1, 2, 2)
    # 画散点
    plt.scatter(X, y, c=y, cmap='bwr', edgecolor='k', s=100, label='Students')
    # 画S型拟合曲线
    X_test = np.linspace(0, 6, 100).reshape(-1, 1)
    y_test_prob = model.predict_proba(X_test)
    plt.plot(X_test, y_test_prob, color='green', linewidth=3, label='Logistic Curve')
    # 画分界线
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.xlabel('Study Hours')
    plt.ylabel('Pass Probability (0-1)')
    plt.title('Study Hours vs. Pass Probability')
    plt.legend()
    
    plt.show()