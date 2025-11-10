import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob

# 檢查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 50)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名稱: {torch.cuda.get_device_name(0)}")
    print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"使用設備: {device}")
print("=" * 50)

# 設定參數
IMG_SIZE = 512
BATCH_SIZE = 16  # 可以根據GPU記憶體調整
EPOCHS = 50
NUM_CLASSES = 4
LEARNING_RATE = 0.001

# 資料夾路徑
BASE_DIR = 'data/generated_images'
DATA_FOLDERS = [os.path.join(BASE_DIR, f'{c}') for c in ['back', 'fetal', 'side', 'stomach']]
CLASS_NAMES = ['A', 'B', 'C', 'D']

class SleepPoseDataset(Dataset):
    """自定義資料集類別"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 讀取圖片
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # 應用轉換
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_image_paths(data_folders):
    """載入所有圖片路徑和標籤"""
    image_paths = []
    labels = []
    
    for class_idx, folder in enumerate(data_folders):
        print(f"掃描 {folder} 的圖片...")
        
        # 取得所有圖片檔案路徑
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(folder, pattern)))
            files.extend(glob.glob(os.path.join(folder, pattern.upper())))
        
        print(f"  找到 {len(files)} 張圖片")
        
        for img_path in files:
            image_paths.append(img_path)
            labels.append(class_idx)
    
    return image_paths, labels

class CNNModel(nn.Module):
    """CNN模型架構"""
    def __init__(self, num_classes=4):
        super(CNNModel, self).__init__()
        
        # 卷積層
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 計算展平後的大小
        # 512 -> 256 -> 128 -> 64 -> 32
        self.flatten_size = 256 * 32 * 32
        
        # 全連接層
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    """訓練一個epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向傳播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        # 統計
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新進度條
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """驗證模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def main():
    # 載入圖片路徑
    print("\n開始載入資料...")
    image_paths, labels = load_image_paths(DATA_FOLDERS)
    print(f"總共 {len(image_paths)} 張圖片")
    print(f"各類別數量: {[labels.count(i) for i in range(NUM_CLASSES)]}")
    
    # 分割訓練集和測試集
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"訓練集大小: {len(train_paths)}")
    print(f"測試集大小: {len(test_paths)}")
    
    # 資料增強和轉換
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 建立資料集和DataLoader
    train_dataset = SleepPoseDataset(train_paths, train_labels, train_transform)
    test_dataset = SleepPoseDataset(test_paths, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # 建立模型
    print("\n建立模型...")
    model = CNNModel(num_classes=NUM_CLASSES).to(device)
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    # 訓練模型
    print("\n開始訓練...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # 訓練
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 驗證
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # 調整學習率
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')
            print(f"✓ 保存最佳模型 (準確率: {best_acc:.2f}%)")
        
        # 清理GPU記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n訓練完成! 最佳驗證準確率: {best_acc:.2f}%")
    
    # 保存最終模型
    torch.save(model.state_dict(), 'final_model.pth')
    print("模型已儲存為 final_model.pth")
    
    return model

if __name__ == "__main__":
    main()