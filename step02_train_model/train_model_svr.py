import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from joblib import dump

# Load Data
using = np.load("../data/train/using_small.npy")
not_using = np.load("../data/train/not_using_small.npy")

# TODO
np.random.shuffle(using)
using = using[:len(not_using) // 2 ]

# Pre Process
X = np.vstack((using, not_using))
y = np.hstack((np.ones(len(using)), np.zeros(len(not_using))))

# Divide
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1919810)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVR Model
svr = SVR(kernel='rbf', C=5, gamma=0.0095)

# Train SVR Model
svr.fit(X_train, y_train)

# Prediction
y_pred_train = svr.predict(X_train)
y_pred_test = svr.predict(X_test)

# Evaluation
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"Train MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")
print(f"Difference: {mse_train-mse_test:.4f}")

# # 可视化训练误差
# plt.figure(figsize=(10, 6))
# plt.scatter(range(len(y_train)), y_train, color='blue', label='Actual')
# plt.scatter(range(len(y_train)), y_pred_train, color='red', label='Predicted')
# plt.title('Training Performance')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

dump(svr, '../data/models/posture_svr.joblib')
dump(scaler, '../data/models/posture_svr_scaler.joblib')