import torch
from torchgeo.datasets import LandCoverAI100
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

device = "mps" if torch.backends.mps.is_available() else "cpu"

CLASS_NAMES = ["Bg", "Building", "Woodland", "Water", "Road"]
CLASS_COLORS = ["#d9d9d9", "#e63946", "#2d6a4f", "#1d70b8", "#f4a261"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(sample):
    img = (sample["image"].float() / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return {"image": img, "mask": sample["mask"].long()}


val_ds = LandCoverAI100(root="data/lc100", split="val", download=False, transforms=preprocess)

# Load both best models
unet = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=5).to(device)
unet.load_state_dict(torch.load("unet_best.pth", map_location=device))
unet.eval()

segformer = smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=5).to(device)
segformer.load_state_dict(torch.load("segformer_best.pth", map_location=device))
segformer.eval()

# Pick the 3 most class-diverse validation tiles
tile_scores = [(len(val_ds[i]["mask"].unique()), i) for i in range(len(val_ds))]
top_indices = [idx for _, idx in sorted(tile_scores, reverse=True)[:3]]

custom_cmap = ListedColormap(CLASS_COLORS)
fig, axes = plt.subplots(3, 4, figsize=(18, 12))

for row, idx in enumerate(top_indices):
    sample = val_ds[idx]
    img = sample["image"].unsqueeze(0).to(device)
    gt_mask = sample["mask"].numpy()

    vis_img = (sample["image"] * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

    with torch.no_grad():
        unet_pred = unet(img).argmax(dim=1).squeeze(0).cpu().numpy()
        seg_pred = segformer(img).argmax(dim=1).squeeze(0).cpu().numpy()

    axes[row, 0].imshow(vis_img)
    axes[row, 0].set_title(f"Tile {idx} — Satellite Input")
    axes[row, 1].imshow(gt_mask, cmap=custom_cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[row, 1].set_title("Ground Truth")
    axes[row, 2].imshow(unet_pred, cmap=custom_cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[row, 2].set_title("U-Net (ResNet-18) — 0.4489 mIoU")
    axes[row, 3].imshow(seg_pred, cmap=custom_cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[row, 3].set_title("SegFormer (MiT-B0) — 0.5384 mIoU")

    for col in range(4):
        axes[row, col].axis("off")

legend_patches = [
    mpatches.Patch(color=color, label=f"{i}: {name}")
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
]
fig.legend(handles=legend_patches, loc="upper center", ncol=5, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.99))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("comparison.png", dpi=150)
print("Saved comparison.png!")