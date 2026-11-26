import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchgeo.datasets import LandCoverAI100
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# 1. Device & constants
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

CLASS_NAMES = ["Bg", "Building", "Woodland", "Water", "Road"]
CLASS_COLORS = ["#d9d9d9", "#e63946", "#2d6a4f", "#1d70b8", "#f4a261"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# 2. Preprocessing: 8x Augmentation (train only) + ImageNet Normalization
def preprocess(sample, augment=False):
    img = sample["image"].float() / 255.0
    mask = sample["mask"].long()

    if augment:
        if random.random() < 0.5:
            img = torch.flip(img, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])

    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return {"image": img, "mask": mask}


# 3. Load LandCoverAI100
train_ds = LandCoverAI100(
    root="data/lc100",
    split="train",
    download=True,
    transforms=lambda s: preprocess(s, augment=True),
)
val_ds = LandCoverAI100(
    root="data/lc100",
    split="val",
    download=True,
    transforms=lambda s: preprocess(s, augment=False),
)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

# 4. Pretrained SegFormer (MiT-B0 Vision Transformer) + Weighted Hybrid Loss
model = smp.Segformer(
    encoder_name="mit_b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=5,
).to(device)

class_weights = torch.tensor([0.5, 3.0, 1.0, 2.5, 2.5], device=device)
ce_loss = nn.CrossEntropyLoss(weight=class_weights)
dice_loss = smp.losses.DiceLoss(mode="multiclass")


def criterion(outputs, masks):
    return 0.5 * ce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)


epochs = 20
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

iou_metric = MulticlassJaccardIndex(num_classes=5, average="none")
best_miou = 0.0
best_class_ious = None

# 5. Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    iou_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()

            preds = outputs.argmax(dim=1)
            correct += (preds == masks).sum().item()
            total += masks.numel()
            iou_metric.update(preds.cpu(), masks.cpu())

    per_class_iou = iou_metric.compute()
    val_miou = per_class_iou.mean().item()
    val_acc = 100 * correct / total

    saved_tag = ""
    if val_miou > best_miou:
        best_miou = val_miou
        best_class_ious = per_class_iou.clone()
        torch.save(model.state_dict(), "segformer_best.pth")
        saved_tag = "  <-- saved best!"

    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | "
        f"Val Loss: {val_loss/len(val_loader):.4f} | "
        f"Val Acc: {val_acc:.1f}% | "
        f"Val mIoU: {val_miou:.4f}{saved_tag}"
    )

# 6. Per-class IoU breakdown
print(f"\nBest SegFormer Val mIoU: {best_miou:.4f}")
print("Per-Class IoU (Best Epoch):")
for name, score in zip(CLASS_NAMES, best_class_ious):
    print(f"  {name:<10}: {score.item():.4f}")

# 7. Plot top 3 most diverse validation tiles with intuitive colors + legend
model.load_state_dict(torch.load("segformer_best.pth", map_location=device))
model.eval()

custom_cmap = ListedColormap(CLASS_COLORS)
tile_scores = [(len(val_ds[i]["mask"].unique()), i) for i in range(len(val_ds))]
top_indices = [idx for _, idx in sorted(tile_scores, reverse=True)[:3]]

fig, axes = plt.subplots(3, 3, figsize=(15, 13))
for row, idx in enumerate(top_indices):
    sample = val_ds[idx]
    img = sample["image"].unsqueeze(0).to(device)
    gt_mask = sample["mask"].numpy()

    vis_img = (
        (sample["image"] * IMAGENET_STD + IMAGENET_MEAN)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    with torch.no_grad():
        pred_mask = model(img).argmax(dim=1).squeeze(0).cpu().numpy()

    axes[row, 0].imshow(vis_img)
    axes[row, 0].set_title(f"Tile {idx} — Satellite Input")
    axes[row, 1].imshow(gt_mask, cmap=custom_cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[row, 1].set_title(f"Ground Truth (classes: {sample['mask'].unique().tolist()})")
    axes[row, 2].imshow(pred_mask, cmap=custom_cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[row, 2].set_title("Best SegFormer Prediction")
    for col in range(3):
        axes[row, col].axis("off")

legend_patches = [
    mpatches.Patch(color=color, label=f"{i}: {name}")
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
]
fig.legend(
    handles=legend_patches,
    loc="upper center",
    ncol=5,
    fontsize=12,
    frameon=True,
    bbox_to_anchor=(0.5, 0.99),
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("prediction_segformer.png", dpi=150)
print("\nSaved prediction_segformer.png!")