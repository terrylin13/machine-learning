# Dependence

```shell
pip install scikit-learn
```

# Usage 
- 引入数据预处理模块
from sklearn.preprocessing import MinMaxScaler, StandardScaler

- 引入模型选择和评估模块
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

- 引入不同的机器学习算法
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

- 引入聚类算法
from sklearn.cluster import KMeans

- 引入降维算法
from sklearn.decomposition import PCA
