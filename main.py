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
import matplotlib.pyplot as plt
import cv2

# 檢查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 設定參數
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 30
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
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_image_paths(data_folders):
    """載入所有圖片路徑和標籤"""
    image_paths = []
    labels = []
    
    for class_idx, folder in enumerate(data_folders):
        print(f"掃描 {folder} 的圖片...")
        
        files = glob.glob(os.path.join(folder, '*.png'))
        
        print(f"  找到 {len(files)} 張圖片")
        
        for img_path in files:
            image_paths.append(img_path)
            labels.append(class_idx)
    
    return image_paths, labels

class CNNModel(nn.Module):
    """CNN模型架構"""
    def __init__(self, num_classes=4):
        super(CNNModel, self).__init__()
        
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
        
        self.flatten_size = 256 * 32 * 32
        
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
        
        # 用於Grad-CAM的鉤子
        self.gradients = None
        self.activations = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 保存最後一層卷積的激活
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        self.activations = x
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def save_gradient(self, grad):
        """保存梯度"""
        self.gradients = grad
    
    def get_activations_gradient(self):
        """獲取激活的梯度"""
        return self.gradients
    
    def get_activations(self):
        """獲取激活值"""
        return self.activations

class GradCAM:
    """Grad-CAM實現"""
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def generate_cam(self, input_image, target_class=None):
        """
        生成Grad-CAM熱力圖
        
        Args:
            input_image: 輸入圖片張量 (1, C, H, W)
            target_class: 目標類別，如果為None則使用預測類別
        
        Returns:
            cam: 熱力圖
            prediction: 預測類別
        """
        # 前向傳播
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向傳播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 獲取梯度和激活
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()
        
        # 計算權重（全局平均池化）
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 加權組合
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        
        # 正規化到0-1
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class
    
    def visualize_cam(self, original_image, cam, alpha=0.5):
        """
        視覺化Grad-CAM
        
        Args:
            original_image: 原始圖片 (PIL Image 或 numpy array)
            cam: 熱力圖
            alpha: 疊加透明度
        
        Returns:
            疊加後的圖片
        """
        # 轉換原始圖片為numpy array
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # 調整熱力圖大小以匹配原始圖片
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # 轉換為彩色熱力圖
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 疊加
        superimposed = heatmap * alpha + original_image * (1 - alpha)
        superimposed = np.uint8(superimposed)
        
        return superimposed, heatmap

def generate_gradcam_examples(model, test_dataset, num_examples=5, save_dir='gradcam_results'):
    """
    生成多個Grad-CAM範例
    
    Args:
        model: 訓練好的模型
        test_dataset: 測試資料集
        num_examples: 生成範例數量
        save_dir: 儲存目錄
    """
    os.makedirs(save_dir, exist_ok=True)
    
    gradcam = GradCAM(model)
    model.eval()
    
    # 反正規化轉換
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    print(f"\n生成 {num_examples} 個Grad-CAM視覺化範例...")
    
    for idx in range(min(num_examples, len(test_dataset))):
        # 獲取測試圖片
        img_tensor, true_label = test_dataset[idx]
        img_path = test_dataset.image_paths[idx]
        
        # 讀取原始圖片
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize((IMG_SIZE, IMG_SIZE))
        
        # 準備輸入
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 生成Grad-CAM
        cam, pred_class = gradcam.generate_cam(input_tensor)
        
        # 反正規化用於顯示
        display_img = inv_normalize(img_tensor)
        display_img = display_img.permute(1, 2, 0).cpu().numpy()
        display_img = np.clip(display_img, 0, 1)
        display_img = (display_img * 255).astype(np.uint8)
        
        # 視覺化
        superimposed, heatmap = gradcam.visualize_cam(original_img, cam)
        
        # 繪製結果
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original Image\nTrue: {CLASS_NAMES[true_label]}')
        axes[0].axis('off')
        
        axes[1].imshow(display_img)
        axes[1].set_title('Normalized Input')
        axes[1].axis('off')
        
        axes[2].imshow(heatmap)
        axes[2].set_title('Grad-CAM Heatmap')
        axes[2].axis('off')
        
        axes[3].imshow(superimposed)
        axes[3].set_title(f'Overlay\nPred: {CLASS_NAMES[pred_class]}')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # 儲存
        save_path = os.path.join(save_dir, f'gradcam_example_{idx+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [{idx+1}/{num_examples}] 已儲存: {save_path}")
        print(f"    真實標籤: {CLASS_NAMES[true_label]}, 預測標籤: {CLASS_NAMES[pred_class]}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    """訓練一個epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
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
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_model.pth')
            print(f"✓ 保存最佳模型 (準確率: {best_acc:.2f}%)")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n訓練完成! 最佳驗證準確率: {best_acc:.2f}%")
    
    # 保存最終模型
    torch.save(model.state_dict(), 'final_model.pth')
    print("模型已儲存為 final_model.pth")
    
    # 生成Grad-CAM視覺化
    print("\n" + "="*50)
    print("生成 Grad-CAM 視覺化...")
    print("="*50)
    generate_gradcam_examples(model, test_dataset, num_examples=10)
    
    return model

if __name__ == "__main__":
    main()