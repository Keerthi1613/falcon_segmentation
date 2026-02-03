import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_dataset import SegmentationDataset
from model import SimpleSegNet   # or UNet if you prefer

# ================= PATHS =================

TRAIN_IMG = "../Offroad_Segmentation_Training_Dataset/train/Color_Images"
TRAIN_MASK = "../Offroad_Segmentation_Training_Dataset/train/Segmentation"

VAL_IMG = "../Offroad_Segmentation_Training_Dataset/val/Color_Images"
VAL_MASK = "../Offroad_Segmentation_Training_Dataset/val/Segmentation"

# ================= DEVICE =================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= TRANSFORMS =================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor() 
])

# ================= DATASETS =================

train_dataset = SegmentationDataset(
    image_dir=TRAIN_IMG,
    mask_dir=TRAIN_MASK,
    transform=transform
)

val_dataset = SegmentationDataset(
    image_dir=VAL_IMG,
    mask_dir=VAL_MASK,
    transform=transform
)

print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))

# ================= DATALOADERS =================

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("Dataloader ready")

# ================= MODEL =================

model = SimpleSegNet(num_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ================= IOU FUNCTION =================

def compute_iou(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection

    if union == 0:
        return torch.tensor(1.0)

    return intersection / union

# ================= TRAINING =================

model.train()
num_epochs = 5   # keep 1 for now

for epoch in range(num_epochs):
    epoch_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(train_loader):.4f}")

print("Training completed")


# ================= VALIDATION (IoU) =================

model.eval()
iou_total = 0
count = 0

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        iou = compute_iou(outputs, masks)

        iou_total += iou.item()
        count += 1

final_iou = iou_total / count
print(f"\n Final Validation IoU: {final_iou:.4f}")
torch.save(model.state_dict(), "segmentation_model.pth")
print("Model saved as segmentation_model.pth")