import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================
# STEP 1: Cleaned Dataset Class
# ======================================
class IndoorDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Select sensor columns
        self.sensor_cols = [c for c in self.data.columns if 'WiFi' in c or 'Geomag' in c or 'Gyro' in c or 'Acc' in c]

        # Convert all to numeric & clean NaN/infinite values
        self.data[self.sensor_cols] = self.data[self.sensor_cols].apply(pd.to_numeric, errors='coerce')
        self.data[self.sensor_cols] = self.data[self.sensor_cols].fillna(0).replace([np.inf, -np.inf], 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['Image_Path']

        # Safe image loading
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))

        if self.transform:
            img = self.transform(img)

        sensors = torch.tensor(row[self.sensor_cols].values.astype(np.float32))
        label = torch.tensor(int(row['Next_Room']) - 1, dtype=torch.long)
        return img, sensors, label

# ======================================
# STEP 2: Load Data
# ======================================
csv_path = r"E:\DATA MINING dataset\new data\synthetic_indoor_localization_data_with_images.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = IndoorDataset(csv_path, transform)

# Train/Validation/Test Split (60/20/20)
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)
test_loader = DataLoader(test_set, batch_size=16)

# ======================================
# STEP 3: HCT Fusion Model
# ======================================
class HCTFusion(nn.Module):
    def __init__(self, num_classes, sensor_dim):
        super().__init__()

        # CNN (ResNet18 without pretrained weights)
        base_cnn = models.resnet18(weights=None)
        base_cnn.fc = nn.Identity()
        self.cnn = base_cnn

        # Sensor encoder
        self.sensor_fc = nn.Sequential(
            nn.Linear(sensor_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Transformer Fusion
        self.query_proj = nn.Linear(512, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, sensor):
        img_feat = self.cnn(img)               # (B, 512)
        sens_feat = self.sensor_fc(sensor)     # (B, 128)

        img_query = self.query_proj(img_feat).unsqueeze(1)
        sens_seq = sens_feat.unsqueeze(1)
        fusion = torch.cat([img_query, sens_seq], dim=1)

        trans_out = self.transformer(fusion)
        combined = torch.cat((trans_out[:, 0, :], sens_feat), dim=1)
        out = self.fc(combined)
        return out

# ======================================
# STEP 4: Setup
# ======================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = len(dataset.data["Next_Room"].unique())
sensor_dim = len(dataset.sensor_cols)

model = HCTFusion(num_classes, sensor_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ======================================
# STEP 5: Train or Load Model
# ======================================
mode = "train"  # Change to "train" to train a new model

def train_model(model, train_loader, val_loader, epochs=8):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, sensors, labels in train_loader:
            imgs, sensors, labels = imgs.to(device), sensors.to(device), labels.to(device)
            preds = model(imgs, sensors)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, sensors, labels in val_loader:
                imgs, sensors, labels = imgs.to(device), sensors.to(device), labels.to(device)
                preds = model(imgs, sensors)
                val_loss += criterion(preds, labels).item()

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
    return train_losses, val_losses

if mode == "train":
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=8)
    torch.save(model.state_dict(), "indoor_hctfusion_model.pth")
    np.save("train_losses.npy", train_losses)
    np.save("val_losses.npy", val_losses)
    print("Model and loss data saved successfully.")

elif mode == "load":
    model.load_state_dict(torch.load("indoor_hctfusion_model.pth", map_location=device))
    model.eval()
    train_losses = np.load("train_losses.npy")
    val_losses = np.load("val_losses.npy")
    print("Loaded pre-trained model successfully.")

# ======================================
# STEP 6: Evaluation
# ======================================
model.eval()
actual, predicted = [], []

with torch.no_grad():
    for imgs, sensors, labels in test_loader:
        imgs, sensors = imgs.to(device), sensors.to(device)
        preds = model(imgs, sensors)
        _, pred_labels = torch.max(preds, 1)
        actual.extend(labels.numpy())
        predicted.extend(pred_labels.cpu().numpy())

accuracy = np.mean(np.array(actual) == np.array(predicted)) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# ======================================
# STEP 7: Plots
# ======================================
plt.figure(figsize=(10,5))
plt.plot(actual[:50], label="Actual Next Room", color='green', linewidth=2)
plt.plot(predicted[:50], label="Predicted Next Room", color='blue', linestyle='--')
plt.title(f"Next Room Prediction (Accuracy: {accuracy:.2f}%)")
plt.xlabel("Sample Index")
plt.ylabel("Room ID")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(train_losses, label='Training Loss', color='orange')
plt.plot(val_losses, label='Validation Loss', color='purple')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
