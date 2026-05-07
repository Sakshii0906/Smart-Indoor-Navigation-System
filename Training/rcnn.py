
import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import box_iou

# 🔹 Set image path
image_path = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train/find_image_specific"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted([
            f for f in os.listdir(root) if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.imgs = self.imgs[:10]  # ✅ Limit for testing
        print(f"Total images loaded: {len(self.imgs)}")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # ✅ Dummy bounding box (replace with real annotations later)
        boxes = torch.tensor([[50, 50, 200, 200]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# 🔹 Load Dataset
dataset = CustomDataset(image_path, transforms=F.to_tensor)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# 🔹 Load Pretrained Faster R-CNN and modify head
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# 🔹 Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 🔹 Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 🔹 Training Loop
num_epochs = 2
print("Starting training...\n")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        print(f"Epoch {epoch+1}, Batch {i+1}/{len(data_loader)}")

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f">>> Epoch {epoch+1} finished. Loss: {running_loss:.4f}\n")

# 🔹 Save model
torch.save(model.state_dict(), "faster_rcnn_model.pth")
print("✅ Model saved as faster_rcnn_model.pth")

# 🔹 Accuracy Evaluation (Basic IoU-based)
model.eval()
correct = 0
total = 0

print("Evaluating accuracy on training data...")

with torch.no_grad():
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for pred, target in zip(outputs, targets):
            pred_boxes = pred['boxes'].cpu()
            true_boxes = target['boxes'].cpu()

            if len(pred_boxes) == 0:
                continue

            ious = box_iou(pred_boxes, true_boxes)
            max_iou = ious.max().item()

            if max_iou > 0.5:
                correct += 1
            total += 1

accuracy = correct / total if total > 0 else 0
print(f"✅ Total Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")









































































import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import box_iou

# 🔹 Set image path
image_path = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train/find_image_specific"

# 🔹 Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted([
            f for f in os.listdir(root) if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.imgs = self.imgs[:10]  # ✅ Limit for testing
        print(f"Total images loaded: {len(self.imgs)}")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # ✅ Dummy bounding box (replace with real annotations later)
        boxes = torch.tensor([[50, 50, 200, 200]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# 🔹 Load full dataset
full_dataset = CustomDataset(image_path, transforms=F.to_tensor)

# 🔹 Train-test split (80-20)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# 🔹 Load and modify pretrained Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# 🔹 Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 🔹 Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 🔹 Training Loop
num_epochs = 2
print("Starting training...\n")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(train_loader):
        print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}")

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f">>> Epoch {epoch+1} finished. Loss: {running_loss:.4f}\n")

# 🔹 Save model
torch.save(model.state_dict(), "faster_rcnn_model.pth")
print("✅ Model saved as faster_rcnn_model.pth")

# 🔹 Evaluation on Test Set (IoU-based)
model.eval()
correct = 0
total = 0

print("Evaluating accuracy on test data...")

with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for pred, target in zip(outputs, targets):
            pred_boxes = pred['boxes'].cpu()
            true_boxes = target['boxes'].cpu()

            if len(pred_boxes) == 0:
                continue

            ious = box_iou(pred_boxes, true_boxes)
            max_iou = ious.max().item()

            if max_iou > 0.5:
                correct += 1
            total += 1

accuracy = correct / total if total > 0 else 0
print(f"✅ Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")




