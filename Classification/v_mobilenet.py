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
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
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

def evaluate_valset():
    loss_list = []
    labels_list = []
    preds_list = []
    prob_list = []

    with torch.no_grad():
        for images, labels in val_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            probs = F.softmax(outputs, dim=1)[:, 1]  # Get probability for class 1
            _, preds = torch.max(outputs, 1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()

            loss = criterion(outputs, labels).item()

            loss_list.append(loss)
            labels_list.extend(labels_np)
            preds_list.extend(preds_np)
            prob_list.extend(probs_np)

    log_val = {
        'epoch': epoch,
        'val_loss': np.mean(loss_list),
        'val_accuracy': accuracy_score(labels_list, preds_list),
        'val_precision': precision_score(labels_list, preds_list, average='binary'),
        'val_recall': recall_score(labels_list, preds_list, average='binary'),
        'val_f1-score': f1_score(labels_list, preds_list, average='binary')
    }

    print(f"val_loss:{log_val['val_loss']}, val_accuracy{log_val['val_accuracy']}")

    # ----- ROC Curve -----
    fpr, tpr, _ = roc_curve(labels_list, prob_list)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC (AUC = {:.2f})'.format(roc_auc_score(labels_list, prob_list)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_buf = io.BytesIO()
    plt.savefig(roc_buf, format='png')
    plt.close()
    roc_buf.seek(0)
    roc_image = Image.open(roc_buf)
    wandb.log({"ROC Curve": wandb.Image(roc_image)})

    # ----- Confusion Matrix -----
    cm = confusion_matrix(labels_list, preds_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    cm_buf = io.BytesIO()
    plt.savefig(cm_buf, format='png')
    plt.close()
    cm_buf.seek(0)
    cm_image = Image.open(cm_buf)
    wandb.log({"Confusion Matrix": wandb.Image(cm_image)})

    return log_val

def evaluate_testset():
    loss_list = []
    labels_list = []
    preds_list = []
    prob_list = []

    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            probs = F.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_np = probs.cpu().numpy()

            loss = criterion(outputs, labels).item()

            loss_list.append(loss)
            labels_list.extend(labels_np)
            preds_list.extend(preds_np)
            prob_list.extend(probs_np)

    log_test = {
        'epoch': epoch,
        'test_loss': np.mean(loss_list),
        'test_accuracy': accuracy_score(labels_list, preds_list),
        'test_precision': precision_score(labels_list, preds_list, average='binary'),
        'test_recall': recall_score(labels_list, preds_list, average='binary'),
        'test_f1-score': f1_score(labels_list, preds_list, average='binary')
    }

    print(f"test_loss:{log_test['test_loss']}, test_accuracy:{log_test['test_accuracy']}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels_list, prob_list)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC (AUC = {:.2f})'.format(roc_auc_score(labels_list, prob_list)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend(loc='lower right')
    roc_buf = io.BytesIO()
    plt.savefig(roc_buf, format='png')
    plt.close()
    roc_buf.seek(0)
    roc_image = Image.open(roc_buf)
    wandb.log({"Test ROC Curve": wandb.Image(roc_image)})

    # Confusion Matrix
    cm = confusion_matrix(labels_list, preds_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    cm_buf = io.BytesIO()
    plt.savefig(cm_buf, format='png')
    plt.close()
    cm_buf.seek(0)
    cm_image = Image.open(cm_buf)
    wandb.log({"Test Confusion Matrix": wandb.Image(cm_image)})

    return log_test


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


model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)
# optimizer = optim.Adam(model.parameters())
optimizer = optim.Adam(model.classifier[1].parameters())

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
        old_best_checkpoint_path = 'mobilenet_64best-acc-{:.3f}.pth'.format(best_test_accuracy)
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)

        best_test_accuracy = log_test['val_accuracy']
        new_best_checkpoint_path = 'mobilenet_64best-acc-{:.3f}.pth'.format(log_test['val_accuracy'])
    
        # ✅ Save only the state_dict
        torch.save(model.state_dict(), new_best_checkpoint_path)
    
        print('Saving the optimal model:', new_best_checkpoint_path)

    # if log_test['val_accuracy'] > best_test_accuracy: 

    #     old_best_checkpoint_path = 'mobilenet_64best-{:.3f}.pth'.format(best_test_accuracy)
    #     if os.path.exists(old_best_checkpoint_path):
    #         os.remove(old_best_checkpoint_path)

    #     best_test_accuracy = log_test['val_accuracy']
    #     new_best_checkpoint_path = 'mobilenet_64best-{:.3f}.pth'.format(log_test['val_accuracy'])
    #     torch.save(model, new_best_checkpoint_path)
    #     print('saving the optimal model', 'mobilenet_64best-{:.3f}.pth'.format(best_test_accuracy))
        # best_test_accuracy = log_test['test_accuracy']

df_train_log.to_csv('mobilenet_64trainlog.csv', index=False)
df_val_log.to_csv('mobilenet_64validationlog.csv', index=False)
print("Evaluating final model on test set...")

# # Load the best model based on validation accuracy before testing
# best_model_path = 'mobilenet_64best-{:.3f}.pth'.format(best_test_accuracy)
# model = torch.load(best_model_path)
# model.to(device)

# # Now evaluate the test set using the best model
# log_test_final = evaluate_testset()

best_model_path = 'mobilenet_64best-acc-{:.3f}.pth'.format(best_test_accuracy)

# ✅ Re-instantiate the model and load weights
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Adjust final layer for 2 classes


model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
model.eval()

# Now evaluate the test set using the best model
log_test_final = evaluate_testset()



# === Save final test results to "mobilenet/" folder ===
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

output_dir = "mobilenet"
os.makedirs(output_dir, exist_ok=True)

# Save test metrics
metrics_path = os.path.join(output_dir, "test_metrics.txt")
with open(metrics_path, "w") as f:
    for k, v in log_test_final.items():
        f.write(f"{k}: {v:.4f}\n")

# Re-run predictions to get values for ROC and Confusion Matrix
labels_list = []
preds_list = []
prob_list = []

model.eval()
with torch.no_grad():
    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        probs = F.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)

        labels_list.extend(labels.cpu().numpy())
        preds_list.extend(preds.cpu().numpy())
        prob_list.extend(probs.cpu().numpy())

# Save ROC Curve
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

# Save Confusion Matrix
cm = confusion_matrix(labels_list, preds_list)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# Save final classifier weights
weights = model.classifier[1].weight.data.cpu().numpy()
biases = model.classifier[1].bias.data.cpu().numpy()

df_weights = pd.DataFrame(weights, columns=[f"w_{i}" for i in range(weights.shape[1])])
df_weights["bias"] = biases
df_weights.to_csv(os.path.join(output_dir, "classifier_weights.csv"), index=False)

print(f"✅ Final evaluation artifacts saved to folder: '{output_dir}/'")

# # Plot Train vs Validation Loss
# plt.figure(figsize=(10, 6))

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

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

plt.ylim(0.0, 1.75)

loss_plot_path = os.path.join(output_dir, "train_val_loss.png")
plt.savefig(loss_plot_path)
plt.show()
