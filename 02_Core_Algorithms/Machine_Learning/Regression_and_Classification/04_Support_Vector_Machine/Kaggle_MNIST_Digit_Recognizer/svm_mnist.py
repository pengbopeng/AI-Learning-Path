import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. 加载数据
# ==========================================
# 注意：train.csv 有 42000 行，为了演示速度，我们可以先取前 5000 行跑通流程
# 等代码没问题了，再把 nrows=5000 去掉，跑全量数据
train_df = pd.read_csv('./dataset/train.csv') # 建议先加 nrows=5000 调试
test_df = pd.read_csv('./dataset/test.csv')

print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)

# 把 label 和 像素点 分开
y = train_df['label']
X = train_df.drop('label', axis=1)

# ==========================================
# 2. 数据可视化 (看看手写数字长啥样)
# ==========================================
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # 把 784 个像素变成 28x28 的矩阵才能画图
    img = X.iloc[i].values.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.show()

# ==========================================
# 3. 数据预处理 (SVM 对数值范围非常敏感！)
# ==========================================
# 像素值是 0-255，我们要把它缩放到 0-1 或者 标准化
# 这里使用标准化 (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df) # 此时 test_df 还是 784维

# ==========================================
# 4. 降维 (PCA) - 提速神器
# ==========================================
# SVM 跑 784 维太慢了，我们保留 95% 的信息量，看看需要多少维
pca = PCA(n_components=0.95) 
X_pca = pca.fit_transform(X_scaled)
test_pca = pca.transform(test_scaled)

print(f"降维后特征数量: {X_pca.shape[1]}") 
# 通常会从 784 降到 300 左右，速度提升一倍以上

# ==========================================
# 5. 训练 SVM 模型
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

print("正在训练 SVM (RBF核)... 可能需要几分钟，请耐心等待...")
# C=10: 对错误容忍度低（严厉），适合数字识别这种边界清晰的任务
model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train, y_train)

# 验证集跑分
val_preds = model.predict(X_val)
acc = accuracy_score(y_val, val_preds)
print(f"✅ SVM 验证集准确率: {acc:.2%}")

# ==========================================
# 6. 错误分析 (看看模型把哪个数字认错了？)
# ==========================================
cm = confusion_matrix(y_val, val_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ==========================================
# 7. 生成提交文件
# ==========================================
final_preds = model.predict(test_pca)
submission = pd.DataFrame({
    'ImageId': range(1, len(final_preds) + 1),
    'Label': final_preds
})
submission.to_csv('digit_submission.csv', index=False)
print("🎉 文件已生成: digit_submission.csv")