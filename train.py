# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models

# 參數設定（比較激進）
DATA_DIR = "data/generated_images"
BATCH_SIZE = 32
LR = 1e-4            # 比之前稍微大一點，配合 full fine-tune
NUM_EPOCHS = 25
PATIENCE = 1000        # 不要早停
RANDOM_SEED = 43
NUM_CLASSES = 4
MODEL_PATH = "best_model.pth"
SPLIT_INFO_PATH = "split_info.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform：訓練用較強增強，驗證/測試用乾淨版
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# 先用 eval_transform 建 base_dataset
base_dataset = datasets.ImageFolder(root=DATA_DIR, transform=eval_transform)
class_names = base_dataset.classes
print("Classes:", class_names)
print("Total images:", len(base_dataset))

# 切 Train / Val / Test (70 / 15 / 15)，用固定 seed
torch.manual_seed(RANDOM_SEED)
n_total = len(base_dataset)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_set, val_set, test_set = random_split(
    base_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

split_info = {
    "test_indices": test_set.indices,
    "class_to_idx": base_dataset.class_to_idx,
    "classes": class_names,
}
torch.save(split_info, SPLIT_INFO_PATH)
print(f"Saved split info to {SPLIT_INFO_PATH}")

# 訓練集改用 train_transform（有增強）
train_set.dataset.transform = train_transform
val_set.dataset.transform = eval_transform
test_set.dataset.transform = eval_transform

# 計算 Train set 的類別統計 & 權重
NUM_CLASSES = len(class_names)
class_counts = torch.zeros(NUM_CLASSES)
targets = base_dataset.targets  # 所有圖片的原始標籤

for idx in train_set.indices:
    class_idx = targets[idx]
    class_counts[class_idx] += 1

print("Class counts in train set:", class_counts.tolist())

# 類別權重：樣本越少，權重越大
class_weights = class_counts.sum() / (NUM_CLASSES * class_counts)
print("Class weights:", class_weights.tolist())

# 每個 sample 的抽樣權重（用於 WeightedRandomSampler）
sample_weights = [class_weights[targets[idx]].item() for idx in train_set.indices]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# 建立 ResNet18，全部 fine-tune
print("Using FULLY fine-tuned ResNet18 (all layers trainable).")
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 讓所有參數可更新
for param in resnet.parameters():
    param.requires_grad = True

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = resnet.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LR)

# 訓練 + Early Stopping
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # ----- Training -----
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_train_loss = running_loss / total
    epoch_train_acc = running_corrects.double() / total

    # ----- Validation -----
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * images.size(0)
            val_corrects += torch.sum(preds == labels.data)
            val_total += labels.size(0)

    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_corrects.double() / val_total

    print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # 儲存最佳模型 + early stopping 控制
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  -> New best model saved to {MODEL_PATH}")
    else:
        epochs_no_improve += 1
        print(f"  -> No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training finished.")
