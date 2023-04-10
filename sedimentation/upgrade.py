import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 读取模拟数据
csv_data = """
#... 请在这里插入之前的模拟数据 ...
"""
df = pd.read_csv(pd.StringIO(csv_data), index_col='timestamp', parse_dates=True)

# 创建滑动窗口特征
window_size = 3
df = create_features(df, window_size)

# 划分特征和目标变量
X = df.drop("Xj1", axis=1)
y = df["Xj1"]

# 对特征进行标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用XGBoost模型
model = xgb.XGBRegressor(random_state=42)

# 为GridSearchCV设置超参数范围
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.5, 1.0],
    "colsample_bytree": [0.5, 1.0],
}

# 使用GridSearchCV寻找最佳超参数
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳超参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳超参数训练模型
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 预测和评估
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
