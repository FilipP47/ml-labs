import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


data = pd.read_csv('mini_projekt_poniedzialek/mini_projekt_poniedzialek/data.csv')


features = data.drop(columns=['instant', 'dteday', 'cnt', 'registered', 'casual'])
labels = data['cnt']
labels = np.log1p(labels)


X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=1):
        super(RegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

input_size = X_train.shape[1]
hidden_size1 = 128
hidden_size2 = 64
model = RegressionModel(input_size, hidden_size1, hidden_size2)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {np.sqrt(loss.item()):.4f}')

model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_loss = criterion(val_outputs, y_val_tensor)
    print(f'Validation Loss: {np.sqrt(val_loss.item())}')

eval_data = pd.read_csv('mini_projekt_poniedzialek/mini_projekt_poniedzialek/evaluation_data.csv')

eval_features = eval_data.drop(columns=['instant', 'dteday'], errors='ignore')
eval_scaled = scaler.transform(eval_features)
eval_tensor = torch.tensor(eval_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    predictions = model(eval_tensor).squeeze().numpy()

predictions = np.expm1(predictions)


pd.DataFrame(predictions, columns=['cnt']).to_csv('predictions.csv', index=False)