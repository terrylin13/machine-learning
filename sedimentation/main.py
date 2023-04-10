import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('settlement_data.csv')  # 假设数据文件名为'settlement_data.csv'
settlement_values = data['settlement'].values.reshape(-1, 1)

# 数据预处理
scaler = MinMaxScaler()
scaled_settlement = scaler.fit_transform(settlement_values)

# 准备训练和测试数据
X = scaled_settlement[:-1]  # 输入：往期沉降数据
y = scaled_settlement[1:]   # 输出：下一期沉降数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 神经网络模型
class SettlementPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SettlementPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 超参数
input_size = 1
hidden_size = 10
output_size = 1
learning_rate = 0.01
epochs = 200

# 创建模型
model = SettlementPredictor(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 测试模型
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test Loss: {test_loss.item():.6f}')

# 预测下一期沉降数据
# next_settlement = model(torch.tensor([scaled_settlement[-1]], dtype=torch.float32))
next_settlement = model(torch.tensor(np.array([scaled_settlement[-1]]), dtype=torch.float32))
next_settlement = scaler.inverse_transform(next_settlement.detach().numpy())
print(f'Predicted Next Settlement: {next_settlement[0][0]:.2f}')
