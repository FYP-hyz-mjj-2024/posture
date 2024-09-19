import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from joblib import dump

# 加载数据
using = np.load("../data/train/using_small.npy")
not_using = np.load("../data/train/not_using_small.npy")

# 临时解决方案，确保数据集大小一致
np.random.shuffle(using)
using = using[:len(not_using)]

# 数据预处理
X = np.vstack((using, not_using))
X[:, ::2] /= 180  # 角度字段归一化到[0, 1]区间
y = np.hstack((np.ones(len(using)), np.zeros(len(not_using))))

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=114514)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVR模型
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)

# 训练模型
svr.fit(X_train, y_train)

# 预测
y_pred_train = svr.predict(X_train)
y_pred_test = svr.predict(X_test)

# 评估模型
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"Train MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")

# 可视化训练误差
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual')
plt.scatter(range(len(y_train)), y_pred_train, color='red', label='Predicted')
plt.title('Training Performance')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()

dump(svr, '../data/models/posture_svr.joblib')
dump(scaler, '../data/models/posture_svr_scaler.joblib')