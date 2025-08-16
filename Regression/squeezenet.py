import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="split dataset and augment train set")
parser.add_argument('--batch', type=str, required=True, help='Batch identifier to replace "batch1+2" in file and folder names (e.g., "batch1").')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
args = parser.parse_args() 
batch = args.batch
num_epochs = args.num_epochs

label_name = 'Firmness'

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, image_label_map, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_label_map = image_label_map
        self.image_names = list(self.image_label_map.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx] + ".jpg")  # Add .jpg extension
        
        image = Image.open(img_name).convert('RGB')
        label = self.image_label_map[self.image_names[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create training dataset and data loader
train_image_label_map = np.load(f'dictfile/{batch}_train.npy', allow_pickle=True).item()
train_dataset = CustomDataset(root_dir=f'dataset_split/{batch}_train', image_label_map=train_image_label_map, transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=21)

# Create validation dataset and data loader
val_image_label_map = np.load(f'dictfile/{batch}_validation.npy', allow_pickle=True).item()
val_dataset = CustomDataset(root_dir=f'dataset_split/{batch}_val', image_label_map=val_image_label_map, transform=transform)
val_data_loader = DataLoader(val_dataset, batch_size=32)

# Create testing dataset and data loader
test_image_label_map = np.load(f'dictfile/{batch}_test.npy', allow_pickle=True).item()
test_dataset = CustomDataset(root_dir=f'dataset_split/{batch}_test', image_label_map=test_image_label_map, transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Create mobilenet model and move it to GPU
mobilenet = models.squeezenet1_0(pretrained=True)
# Replace the classifier with a regression head
mobilenet.classifier[1] = nn.Conv2d(512, 1, kernel_size=1)
mobilenet.num_classes = 1  # Optional, for consistency
mobilenet = mobilenet.to(device)
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(mobilenet.parameters(),  lr=0.0001, weight_decay=1e-5)

import matplotlib.pyplot as plt

best_val_loss = float('inf')  # Initialize best validation loss to infinity
best_model_weights = None  # Initialize best model weights to None

train_losses = []
val_losses = []

if not os.path.exists('RegressionWeights'):
    os.mkdir('RegressionWeights')

# Train the model
for epoch in range(num_epochs):
    # Train the model
    mobilenet.train()
    for inputs, targets in train_data_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = mobilenet(inputs)
        outputs = outputs.view(outputs.size(0))  # <- Flatten to [batch]
        #print(outputs)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
    
    # Evaluate the model on validation set
    mobilenet.eval()
    val_loss = 0.0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in val_data_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
            outputs = mobilenet(inputs)
            outputs = outputs.view(outputs.size(0))
            #print("validation output:",outputs)
            all_targets.extend(targets.cpu().numpy())  # Move targets to CPU and convert to NumPy
            all_predictions.extend(outputs.cpu().numpy())  # Move predictions to CPU and convert to NumPy

            val_loss += criterion(outputs.squeeze(), targets.float()).item()
    val_loss /= len(val_data_loader)

    train_losses.append(loss.item())
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()}, Val Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_pkl_path = f'RegressionWeights/{batch}_Best_squeezenet_{num_epochs}.pth'  # Use .pth extension for PyTorch model
        torch.save(mobilenet, model_pkl_path)  # Save the entire model (architecture + weights)
        final_model = model_pkl_path  # Keep track of the saved model path

#model_pkl_path = f'RegressionWeights/{batch}_Best_{num_epochs}.pkl'
#torch.save(mobilenet.state_dict(), f'RegressionWeights/{batch}_Last_{num_epochs}.pth')
#with open(model_pkl_path, 'wb') as f:
#    pickle.dump(mobilenet, f)
mse = mean_squared_error(all_targets, all_predictions)
r2 = r2_score(all_targets, all_predictions)

print(f"Test Mean Squared Error: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")

print(f'Model saved as {final_model}')

# Plot the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
if not os.path.exists('LossFunction'):
    os.mkdir('LossFunction')
from datetime import datetime
date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
plt.savefig(f'LossFunction/SqueezeNet_{date_time}_{batch}_{num_epochs}.png')
plt.close()
# Save training and validation loss values to CSV
loss_df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Train Loss': train_losses,
    'Validation Loss': val_losses
})

if not os.path.exists('LossFunction'):
    os.mkdir('LossFunction')

loss_csv_path = f'LossFunction/Losses_squeezenet_{date_time}_{batch}_{num_epochs}.csv'
loss_df.to_csv(loss_csv_path, index=False)
print(f"Loss values saved to {loss_csv_path}")

# Plot Predicted vs Actual values
plt.figure(figsize=(7, 7))
plt.scatter(all_targets, all_predictions, alpha=0.6, color='dodgerblue', edgecolor='k')
plt.plot([min(all_targets), max(all_targets)],
         [min(all_targets), max(all_targets)],
         color='red', linestyle='--', linewidth=2, label='Ideal Fit (y = x)')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Predicted vs Actual (R² = {r2:.4f})')
plt.legend()
plt.grid(True)

pred_vs_actual_path = f'graphs/Predicted_vs_Actual_squeezeNet{date_time}_{batch}_{num_epochs}.png'
plt.savefig(pred_vs_actual_path)
plt.close()

print(f"Predicted vs Actual plot saved to {pred_vs_actual_path}")

# Load the best saved model
best_model = torch.load(final_model, weights_only=False)

best_model.eval()

# Evaluate on test set
test_loss = 0.0
test_targets = []
test_predictions = []

with torch.no_grad():
    for inputs, targets in test_data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = best_model(inputs)
        test_targets.extend(targets.cpu().numpy())
        test_predictions.extend(outputs.cpu().numpy())
        test_loss += criterion(outputs.squeeze(), targets.float()).item()
test_loss /= len(test_data_loader)

# Compute test metrics
test_mse = mean_squared_error(test_targets, test_predictions)
test_r2 = r2_score(test_targets, test_predictions)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Mean Squared Error: {test_mse:.4f}")
print(f"Test R² Score: {test_r2:.4f}")

# Plot predicted vs actual for test set
plt.figure(figsize=(7, 7))
plt.scatter(test_targets, test_predictions, alpha=0.6, color='limegreen', edgecolor='k')
plt.plot([min(test_targets), max(test_targets)],
         [min(test_targets), max(test_targets)],
         color='red', linestyle='--', linewidth=2, label='Ideal Fit (y = x)')

plt.xlabel('Actual Values (Test)')
plt.ylabel('Predicted Values (Test)')
plt.title(f'Test Predicted vs Actual (R² = {test_r2:.4f})')
plt.legend()
plt.grid(True)

if not os.path.exists('graphs'):
    os.makedirs('graphs')

test_pred_vs_actual_path = f'graphs/Test_Predicted_vs_Actual_SqueezeNet_{date_time}_{batch}_{num_epochs}.png'
plt.savefig(test_pred_vs_actual_path)
plt.close()

print(f"Test Predicted vs Actual plot saved to {test_pred_vs_actual_path}")



# Save R² and RMSE to a text file
metrics_path = f'graphs/Test_Metrics_squeezenet_{date_time}_{batch}_{num_epochs}.txt'

rmse = np.sqrt(test_mse)

with open(metrics_path, 'w') as f:
    f.write(f"Test R² Score: {test_r2:.4f}\n")
    f.write(f"Test RMSE: {rmse:.4f}\n")

print(f"Test metrics saved to {metrics_path}")