import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = Malignant, 1 = Benign

# Ensure Correct Label Mapping (1 = Malignant, 0 = Benign)
y = np.where(y == 0, 1, 0)  # Swap labels to match expected output

# Split Data (Ensure class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as scaler.pkl")

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Verify Label Mapping
print("Unique Labels in Training Data:", np.unique(y_train.cpu().numpy()))
print("Malignant Count:", sum(y_train.cpu().numpy() == 1))
print("Benign Count:", sum(y_train.cpu().numpy() == 0))

# Compute Class Weights for Imbalance Handling
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Define Neural Network with BatchNorm & Dropout
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch Normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Regularization
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size // 4, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

# Initialize Model
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 150

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss & Optimizer
criterion = nn.BCELoss(weight=class_weights[y_train.long()])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Model
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == y_train).float().mean()
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%")

# Save Model
torch.save(model.state_dict(), "breast_cancer_model.pth")
print("✅ Model saved as breast_cancer_model.pth")
