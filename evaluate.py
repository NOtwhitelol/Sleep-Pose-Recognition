import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import cv2

DATA_DIR = "data/generated_images"
REAL_DIR = "data/real_images"
BATCH_SIZE = 32
SPLIT_INFO_PATH = "split_info.pt"
MODEL_PATH = "best_model.pth"
USE_RESNET = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Improved Grad-CAM Implementation (Based on Reference Code) ---
class ImprovedGradCAM:
    """
    改進版 Grad-CAM，採用更精準的激活值和梯度捕獲方式
    參考別人的實現方式
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.model.eval()
    
    def save_gradient(self, grad):
        """保存梯度的回調函數"""
        self.gradients = grad
    
    def forward_with_hook(self, x, target_layer):
        """
        執行 forward pass 並在目標層註冊 hook
        這是關鍵改進：在 forward 過程中動態註冊 hook
        """
        # 前面的層
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        
        # 在 layer3 或 layer4 捕獲激活值
        if target_layer == 'layer3':
            # 在 layer3 輸出時捕獲 (14x14 更高解析度)
            if x.requires_grad:
                x.register_hook(self.save_gradient)
            self.activations = x.clone()
            x = self.model.layer4(x)
        else:
            # 在 layer4 輸出時捕獲 (7x7)
            x = self.model.layer4(x)
            if x.requires_grad:
                x.register_hook(self.save_gradient)
            self.activations = x.clone()
        
        # 全連接層
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        
        return x
    
    def generate_cam(self, input_image, target_class=None, target_layer='layer3'):
        """
        生成 Grad-CAM 熱力圖
        
        Args:
            input_image: 輸入圖片張量 (1, C, H, W)
            target_class: 目標類別，如果為None則使用預測類別
            target_layer: 'layer3' (14x14) 或 'layer4' (7x7)
        
        Returns:
            cam: 熱力圖 (H, W)
            prediction: 預測類別
        """
        self.model.eval()
        input_image.requires_grad = True
        
        # Forward pass with hook registration
        output = self.forward_with_hook(input_image, target_layer)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass using one-hot encoding (更標準的做法)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 獲取梯度和激活值
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # 全局平均池化計算權重
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加權組合
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # ReLU 只保留正值
        
        # 正規化到 [0, 1]
        cam = cam.squeeze().cpu().detach().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class

def apply_colormap_on_image(org_im, activation_map, target_size=(512, 512)):
    """
    Apply heatmap on image with high resolution
    """
    # Resize to target high resolution using CUBIC interpolation
    activation_map_hires = cv2.resize(
        activation_map, 
        target_size, 
        interpolation=cv2.INTER_CUBIC
    )
    
    # Apply Gaussian blur for smoothing
    kernel_size = max(target_size[0] // 100, 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    activation_map_hires = cv2.GaussianBlur(
        activation_map_hires, 
        (kernel_size, kernel_size), 
        0
    )
    
    # Renormalize after blur
    if activation_map_hires.max() > activation_map_hires.min():
        activation_map_hires = (activation_map_hires - activation_map_hires.min()) / \
                               (activation_map_hires.max() - activation_map_hires.min())
    
    # Resize original image to match target size
    org_im_resized = cv2.resize(
        org_im,
        target_size,
        interpolation=cv2.INTER_CUBIC
    )
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map_hires), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    # Superimpose heatmap on image
    cam_image = 0.5 * heatmap + 0.5 * org_im_resized
    cam_image = np.clip(cam_image, 0, 1)
    
    return cam_image, activation_map_hires, org_im_resized

def visualize_gradcam_and_save(model, dataset, num_samples=8, target_layer='layer3', save_dir="gradcam_results"):
    """
    生成每個樣本的 Grad-CAM 可視化，並將原圖 / Grad-CAM / Overlay 與 True / Pred label 放在同一張圖中儲存。
    """
    gradcam = ImprovedGradCAM(model)
    
    print(f"Using {target_layer} for Grad-CAM generation")
    
    # 反標準化
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # 隨機選取樣本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # 創建資料夾
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, sample_idx in enumerate(indices):
        image, label = dataset[sample_idx]
        
        input_tensor = image.unsqueeze(0).to(device)
        cam, pred_class = gradcam.generate_cam(input_tensor, target_layer=target_layer)
        
        # 反標準化原圖
        img_denorm = inv_normalize(image).cpu().numpy()
        img_denorm = np.transpose(img_denorm, (1, 2, 0))
        img_denorm = np.clip(img_denorm, 0, 1)
        
        # Overlay
        cam_image, cam_hires, img_hires = apply_colormap_on_image(img_denorm, cam, target_size=(512, 512))
        
        # --- 使用 Matplotlib 顯示並儲存 ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
        
        axes[0].imshow(np.clip(img_hires, 0, 1))
        axes[0].set_title(f"Original\nTrue: {classes[label]}", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(np.clip(cam_hires, 0, 1), cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f"Grad-CAM\nPred: {classes[pred_class]}", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(np.clip(cam_image, 0, 1))
        axes[2].set_title(f"True: {classes[label]}, Pred: {classes[pred_class]}", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved Grad-CAM figure for sample {idx} at {save_path}")
    
    print("All Grad-CAM figures saved successfully!")


# --- Load split info ---
split_info = torch.load(SPLIT_INFO_PATH)
test_indices = split_info["test_indices"]
class_to_idx = split_info["class_to_idx"]
classes = split_info["classes"]

print("Classes:", classes)
print("class_to_idx:", class_to_idx)
print("Test indices length:", len(test_indices))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Generated data ---
generated_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
test_set_generated = Subset(generated_dataset, test_indices)

# --- Real data ---
real_dataset = datasets.ImageFolder(root=REAL_DIR, transform=transform)

if real_dataset.classes != classes:
    raise ValueError(
        f"ERROR: real_images classes {real_dataset.classes} != original classes {classes}"
    )

print(f"Loaded real_images, total = {len(real_dataset)}")

# --- Combined test set ---
combined_test_set = ConcatDataset([test_set_generated, real_dataset])

test_loader = DataLoader(combined_test_set, batch_size=BATCH_SIZE, shuffle=False)

# --- Build model ---
num_classes = len(classes)

if USE_RESNET:
    print("Using pretrained ResNet18 for evaluation.")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(num_ftrs, num_classes)
    model = resnet.to(device)
else:
    print("Using custom SleepPoseCNN for evaluation.")
    model = SleepPoseCNN(num_classes=num_classes).to(device)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Evaluation
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

cm = confusion_matrix(all_labels, all_preds)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

# ========== Confusion Matrix ==========
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 5))
    # 使用深藍到淺藍的顏色
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()  # 可以加上顏色條
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names, rotation=45)

    # 在格子中標上數字
    thresh = cm.max() / 2.0  # 用來決定文字顏色
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=12)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    print("Confusion matrix saved to confusion_matrix.png")
    plt.show()

plot_confusion_matrix(cm, classes)

# ========== Improved Grad-CAM Visualization ==========
# print("\n" + "="*60)
# print("Generating High-Resolution Grad-CAM visualizations (512x512)...")
# print("Using IMPROVED implementation with dynamic hook registration")
# print("="*60)
# visualize_gradcam_and_save(model, combined_test_set, num_samples=8, target_layer='layer3')

# print("\nEvaluation complete!")