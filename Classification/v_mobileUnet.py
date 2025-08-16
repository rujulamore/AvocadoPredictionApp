import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import io

import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import wandb
wandb.init(project='avocado', name=time.strftime('%m%d%H%M%S'))

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.pointwise(self.depthwise(x))))
        return x

class MobileUNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.enc1 = DepthwiseSeparableConv(3, 32)
        self.enc2 = DepthwiseSeparableConv(32, 64)
        self.enc3 = DepthwiseSeparableConv(64, 128)
        self.enc4 = DepthwiseSeparableConv(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = DepthwiseSeparableConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DepthwiseSeparableConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DepthwiseSeparableConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DepthwiseSeparableConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DepthwiseSeparableConv(64, 32)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        pooled = self.global_pool(d1).view(x.size(0), -1)
        out = self.fc(pooled)
        return out


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
    
def train_one_batch(images, labels):

    images = images.to(device)
    labels = labels.to(device)
    
    outputs = model(images)

    loss = criterion(outputs, labels) 
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    _, preds = torch.max(outputs, 1) 
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx

    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    log_train['train_precision'] = precision_score(labels, preds, average='binary')
    log_train['train_recall'] = recall_score(labels, preds, average='binary')
    log_train['train_f1-score'] = f1_score(labels, preds, average='binary')
    # print(f"train_loss:{loss}, train_accuracy:{log_train['train_accuracy']}")
    return log_train

def evaluate_testset():
    model.eval()
    loss_list = []
    labels_list = []
    preds_list = []
    probs_list = []

    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            probs = F.softmax(outputs, dim=1)[:, 1]  # Class 1 probabilities
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss_list.append(loss.item())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())

    log_test = {}
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='binary')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='binary')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='binary')
    log_test['test_auc'] = roc_auc_score(labels_list, probs_list)

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels_list, probs_list)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % log_test['test_auc'])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend(loc="lower right")
    roc_buf = io.BytesIO()
    plt.savefig(roc_buf, format='png')
    roc_buf.seek(0)
    roc_image = Image.open(roc_buf)
    wandb.log({"Test ROC Curve": wandb.Image(roc_image)})
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(labels_list, preds_list)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Test Confusion Matrix")
    cm_buf = io.BytesIO()
    plt.savefig(cm_buf, format='png')
    cm_buf.seek(0)
    cm_image = Image.open(cm_buf)
    wandb.log({"Test Confusion Matrix": wandb.Image(cm_image)})
    plt.close()

    print(f"Test loss: {log_test['test_loss']:.4f}, Test accuracy: {log_test['test_accuracy']:.4f}")
    return log_test


def evaluate_valset():
    output_dir = "Unet"  # <-- Add this line
    os.makedirs(output_dir, exist_ok=True)  # Create folder if not exists

    loss_list = []
    labels_list = []
    preds_list = []
    probs_list = []

    with torch.no_grad():
        for images, labels in val_data_loader: 
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images) 
            
            probs = F.softmax(outputs, dim=1)[:, 1]  # Get probability of class 1
            _, preds = torch.max(outputs, 1) 
            
            loss = criterion(outputs, labels) 
            
            loss_list.append(loss.item())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())

    log_val = {}
    log_val['epoch'] = epoch
    log_val['val_loss'] = np.mean(loss_list)
    log_val['val_accuracy'] = accuracy_score(labels_list, preds_list)
    log_val['val_precision'] = precision_score(labels_list, preds_list, average='binary')
    log_val['val_recall'] = recall_score(labels_list, preds_list, average='binary')
    log_val['val_f1-score'] = f1_score(labels_list, preds_list, average='binary')

    # ROC Curve and AUC
    # fpr, tpr, _ = roc_curve(labels_list, probs_list)
    # roc_auc = auc(fpr, tpr)
    # log_val['val_auc'] = roc_auc

    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # roc_buf = io.BytesIO()
    # plt.savefig(roc_buf, format='png')
    # roc_buf.seek(0)
    # roc_image = Image.open(roc_buf)
    # wandb.log({"ROC Curve": wandb.Image(roc_image)})

    # plt.close()
    # Save ROC Curve plot (with larger font)
    fpr, tpr, _ = roc_curve(labels_list, probs_list)
    roc_auc = roc_auc_score(labels_list, probs_list)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Test ROC Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()


    # Save Confusion Matrix plot (with larger font)
    cm = confusion_matrix(labels_list, preds_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    
    # Update title and labels with larger fonts
    plt.title("Test Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    print(f"val_loss:{log_val['val_loss']}, val_accuracy{log_val['val_accuracy']}")
    return log_val

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainpath = 'dataset/1400train'
valpath = 'dataset/1400val'

keyvalue_train = 'dictfile/1400train.npy'
keyvalue_val = 'dictfile/1400val.npy'

testpath = 'dataset/1400test'
keyvalue_test = 'dictfile/1400test.npy'

test_image_label_map = np.load(keyvalue_test, allow_pickle=True).item()
test_dataset = CustomDataset(root_dir=testpath, image_label_map=test_image_label_map, transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=64)


# Create training dataset and data loader
train_image_label_map = np.load(keyvalue_train, allow_pickle=True).item()
train_dataset = CustomDataset(root_dir=trainpath, image_label_map=train_image_label_map, transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create validation dataset and data loader
val_image_label_map = np.load(keyvalue_val, allow_pickle=True).item()
val_dataset = CustomDataset(root_dir=valpath, image_label_map=val_image_label_map, transform=transform)
val_data_loader = DataLoader(val_dataset, batch_size=64)


model = MobileUNet(num_classes=2)
optimizer = optim.Adam(model.parameters())


# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model = model.to(device)
criterion = nn.CrossEntropyLoss() 
EPOCHS = 400
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


epoch = 0
batch_idx = 0
best_test_accuracy = 0

df_train_log = pd.DataFrame()
log_train = {}
log_train['epoch'] = 0
log_train['batch'] = 0
images, labels = next(iter(train_data_loader))
log_train.update(train_one_batch(images, labels))
df_train_log = df_train_log._append(log_train, ignore_index=True)

df_val_log = pd.DataFrame()
df_train_log = pd.DataFrame()

for epoch in range(1, EPOCHS+1):
    
    print(f'Epoch {epoch}/{EPOCHS}')
    
    ## train
    model.train()
    for images, labels in tqdm(train_data_loader): 
        batch_idx += 1
        log_train = train_one_batch(images, labels)
        df_train_log = df_train_log._append(log_train, ignore_index=True)
        wandb.log(log_train)
    lr_scheduler.step()

    ## val
    model.eval()
    log_test = evaluate_valset()
    df_val_log = df_val_log._append(log_test, ignore_index=True)
    wandb.log(log_test)
    
    if log_test['val_accuracy'] > best_test_accuracy: 
        old_best_checkpoint_path = 'unet_64best-acc-{:.3f}.pth'.format(best_test_accuracy)
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)

        best_test_accuracy = log_test['val_accuracy']
        new_best_checkpoint_path = 'unet_64best-acc-{:.3f}.pth'.format(log_test['val_accuracy'])
    
        # ✅ Save only the state_dict
        torch.save(model.state_dict(), new_best_checkpoint_path)
    
        print('Saving the optimal model:', new_best_checkpoint_path)


    # if log_test['val_accuracy'] > best_test_accuracy: 

    #     old_best_checkpoint_path = 'unet_64best-{:.3f}.pth'.format(best_test_accuracy)
    #     if os.path.exists(old_best_checkpoint_path):
    #         os.remove(old_best_checkpoint_path)

    #     best_test_accuracy = log_test['val_accuracy']
    #     new_best_checkpoint_path = 'unet_64best-{:.3f}.pth'.format(log_test['val_accuracy'])
    #     torch.save(model, new_best_checkpoint_path)
    #     print('saving the optimal model', 'unet_64best-{:.3f}.pth'.format(best_test_accuracy))
        # best_test_accuracy = log_test['test_accuracy']

df_train_log.to_csv('unet_64trainlog.csv', index=False)
df_val_log.to_csv('unet_64validationlog.csv', index=False)

# # Load the best model based on validation accuracy before testing
# best_model_path = 'unet_64best-{:.3f}.pth'.format(best_test_accuracy)
# model = torch.load(best_model_path)
# model.to(device)

# # Now evaluate the test set using the best model
# log_test_final = evaluate_testset()

# Load the best model weights before testing
best_model_path = 'unet_64best-acc-{:.3f}.pth'.format(best_test_accuracy)

# ✅ Re-instantiate the model and load weights
model = MobileUNet(num_classes=2)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

# Now evaluate the test set using the best model
log_test_final = evaluate_testset()

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

output_dir = "Unet"
os.makedirs(output_dir, exist_ok=True)

# Save test metrics to a txt file
metrics_path = os.path.join(output_dir, "test_metrics.txt")
with open(metrics_path, "w") as f:
    for k, v in log_test_final.items():
        f.write(f"{k}: {v:.4f}\n")

# Re-run predictions to get labels, preds, and probs for ROC and confusion matrix
labels_list = []
preds_list = []
prob_list = []

model.eval()
with torch.no_grad():
    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        probs = F.softmax(outputs, dim=1)[:, 1]  # probability of class 1
        _, preds = torch.max(outputs, 1)

        labels_list.extend(labels.cpu().numpy())
        preds_list.extend(preds.cpu().numpy())
        prob_list.extend(probs.cpu().numpy())

# Save ROC Curve plot
fpr, tpr, _ = roc_curve(labels_list, prob_list)
roc_auc = roc_auc_score(labels_list, prob_list)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# Save Confusion Matrix plot
cm = confusion_matrix(labels_list, preds_list)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# Save classifier weights and biases from MobileUNet.fc layer
weights = model.fc.weight.data.cpu().numpy()
biases = model.fc.bias.data.cpu().numpy()

df_weights = pd.DataFrame(weights, columns=[f"w_{i}" for i in range(weights.shape[1])])
df_weights["bias"] = biases
df_weights.to_csv(os.path.join(output_dir, "classifier_weights.csv"), index=False)

print(f"✅ Final evaluation artifacts saved to folder: '{output_dir}/'")

# Plot Train vs Validation Loss
plt.figure(figsize=(10, 6))

# # Group by epoch and average train loss (since you log per batch)
# train_loss_per_epoch = df_train_log.groupby('epoch')['train_loss'].mean()
# val_loss_per_epoch = df_val_log.set_index('epoch')['val_loss']

# plt.plot(train_loss_per_epoch.index, train_loss_per_epoch.values, label='Train Loss', marker='o')
# plt.plot(val_loss_per_epoch.index, val_loss_per_epoch.values, label='Validation Loss', marker='x')

# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training vs Validation Loss')
# plt.legend()
# plt.grid(True)

# loss_plot_path = os.path.join(output_dir, "train_val_loss.png")
# plt.savefig(loss_plot_path)
# plt.show()
# Plot Train vs Validation Loss
plt.figure(figsize=(10, 6))

# Group by epoch and average train loss (since you log per batch)
train_loss_per_epoch = df_train_log.groupby('epoch')['train_loss'].mean()
val_loss_per_epoch = df_val_log.set_index('epoch')['val_loss']

plt.plot(train_loss_per_epoch.index, train_loss_per_epoch.values, label='Train Loss', marker='o')
plt.plot(val_loss_per_epoch.index, val_loss_per_epoch.values, label='Validation Loss', marker='x')

# Font-enhanced labeling
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training vs Validation Loss', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save and display
loss_plot_path = os.path.join(output_dir, "train_val_loss.png")
plt.tight_layout()
plt.savefig(loss_plot_path)
plt.show()
