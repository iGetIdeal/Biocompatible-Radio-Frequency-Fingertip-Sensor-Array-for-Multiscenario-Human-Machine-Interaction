# 训练BP网络，使用PyTorch实现
# 对频率、回波损耗强弱进行二特征分类
# 训练集：train_augmented_500.xlsx

import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix, roc_curve, auc  
import matplotlib.pyplot as plt  
import seaborn as sns  
import numpy as np  
import pandas as pd  
from itertools import cycle  

# --- 读数据 ---  
excel_path = r'D:\HuaweiMoveData\Users\gzq11\Desktop\人机交互\实验\机器学习\频率-强度双标签\train_augmented_500.xlsx'  # 替换成你的路径  
df = pd.read_excel(excel_path)  

X = df[['Frequency', 'Returnloss']].values  
y = df['label'].values  

# 特征标准化  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# 标签编码  
le = LabelEncoder()  
y_encoded = le.fit_transform(y)  
num_classes = len(le.classes_)  

# 划分数据  
X_train, X_test, y_train, y_test = train_test_split(  
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)  

# 转tensor  
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  
y_test_tensor = torch.tensor(y_test, dtype=torch.long)  

# --- BP模型 ---  
class BPNet(nn.Module):  
    def __init__(self):  
        super(BPNet, self).__init__()  
        self.fc1 = nn.Linear(2, 16)  
        self.fc2 = nn.Linear(16, 16)  
        self.fc3 = nn.Linear(16, num_classes)  
        self.relu = nn.ReLU()  
    
    def forward(self, x):  
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x  

model = BPNet()  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  

# --- 训练 ---  
epochs = 50  
batch_size = 8  

for epoch in range(epochs):  
    model.train()  
    permutation = torch.randperm(X_train_tensor.size()[0])  
    epoch_loss = 0  
    
    for i in range(0, X_train_tensor.size()[0], batch_size):  
        indices = permutation[i:i+batch_size]  
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]  
        optimizer.zero_grad()  
        outputs = model(batch_x)  
        loss = criterion(outputs, batch_y)  
        loss.backward()  
        optimizer.step()  
        epoch_loss += loss.item()  
    if epoch % 10 == 0 or epoch == epochs - 1:  
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss / (X_train_tensor.size(0) // batch_size):.4f}")  

# --- 得到训练集和测试集的预测 ---  
model.eval()  
with torch.no_grad():  
    train_outputs = model(X_train_tensor)  
    test_outputs = model(X_test_tensor)  
    _, train_preds = torch.max(train_outputs, 1)  
    _, test_preds = torch.max(test_outputs, 1)  

train_labels = y_train  
test_labels = y_test  

# --- 混淆矩阵函数 ---  
def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix'):  
    cm = confusion_matrix(y_true, y_pred)  
    plt.figure(figsize=(10,10))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)  
    plt.ylabel('True label')  
    plt.xlabel('Predicted label')  
    plt.title(title)  
    plt.show()  

plot_confusion_matrix(train_labels, train_preds.numpy(), le.classes_, title="Training Set Confusion Matrix")  
plot_confusion_matrix(test_labels, test_preds.numpy(), le.classes_, title="Test Set Confusion Matrix")  

# --- ROC曲线 (One-vs-Rest) ---  
from sklearn.preprocessing import label_binarize  

# 标签二值化  
y_test_bin = label_binarize(test_labels, classes=range(num_classes))  
softmax = nn.Softmax(dim=1)  
test_probs = softmax(test_outputs).numpy()  

plt.figure(figsize=(10, 10))  
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])  
for i, color in zip(range(num_classes), colors):  
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs[:, i])  
    roc_auc = auc(fpr, tpr)  
    plt.plot(fpr, tpr, color=color, lw=2,  
             label=f'Class {le.classes_[i]} (AUC = {roc_auc:.2f})')  

plt.plot([0, 1], [0, 1], 'k--', lw=2)  
plt.xlim([-0.02, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Multiclass ROC Curve')  
plt.legend(loc='lower right')  
plt.show()  

# --- 分类边界绘制 ---  
# 仅在二维特征空间画决策边界  
h = 0.01  # 网格步长  
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1  
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1  
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  
                     np.arange(y_min, y_max, h))  
grid = np.c_[xx.ravel(), yy.ravel()]  
grid_tensor = torch.tensor(grid, dtype=torch.float32)  

model.eval()  
with torch.no_grad():  
    Z = model(grid_tensor)  
    Z = torch.argmax(Z, axis=1).numpy()  
Z = Z.reshape(xx.shape)  

plt.figure(figsize=(10, 10))  
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.get_cmap('tab10'))  

# 标出训练样本点  
for i, class_label in enumerate(le.classes_):  
    idx = y_train == i  
    plt.scatter(X_train[idx, 0], X_train[idx, 1], label=class_label, edgecolor='k', s=20)  
plt.xlabel('Frequency (standardized)')  
plt.ylabel('Returnloss (standardized)')  
plt.title('Decision Boundary and Training Points')  
plt.legend()  
plt.show()  

# 训练集散点图（真实标签）  
plt.figure(figsize=(8,8))  
for i, cls in enumerate(le.classes_):  
    idx = y_train == i  
    plt.scatter(X_train[idx, 0], X_train[idx, 1], label=f"{cls}", alpha=0.7, edgecolors='k', s=50)  
plt.title("Train Set Scatter Plot (True Labels)")  
plt.xlabel("Frequency (standardized)")  
plt.ylabel("Returnloss (standardized)")  
plt.legend()  
plt.show()  

# 测试集散点图（真实标签）  
plt.figure(figsize=(8,8))  
for i, cls in enumerate(le.classes_):  
    idx = y_test == i  
    plt.scatter(X_test[idx, 0], X_test[idx, 1], label=f"{cls}", alpha=0.7, edgecolors='k', s=50)  
plt.title("Test Set Scatter Plot (True Labels)")  
plt.xlabel("Frequency (standardized)")  
plt.ylabel("Returnloss (standardized)")  
plt.legend()  
plt.show()  

# --- 计算准确率 ---  
from sklearn.metrics import accuracy_score  

train_acc = accuracy_score(train_labels, train_preds.numpy())  
test_acc = accuracy_score(test_labels, test_preds.numpy())  

print(f"Training Accuracy: {train_acc:.4f}")  
print(f"Test Accuracy: {test_acc:.4f}")  

# --- 所有数据散点图（真实标签） ---  
plt.figure(figsize=(8,8))  
for i, cls in enumerate(le.classes_):  
    idx = y_encoded == i  
    plt.scatter(X_scaled[idx, 0], X_scaled[idx, 1], label=f"{cls}", alpha=0.7, edgecolors='k', s=50)  
plt.title("All Data Scatter Plot (True Labels)")  
plt.xlabel("Frequency (standardized)")  
plt.ylabel("Returnloss (standardized)")  
plt.legend()  
plt.show()  