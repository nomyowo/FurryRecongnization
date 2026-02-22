from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.array(img)


def to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    return mask


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)


@lru_cache(maxsize=1)
def load_yolo_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


@lru_cache(maxsize=1)
def load_resnet_model():
    """Load ResNet18 model for feature extraction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Remove the final classification layer (fc)
    # ResNet structure: (conv1) ... (avgpool) -> (fc)
    # We want features from avgpool (512-dim)
    model.fc = nn.Identity()

    model.to(device)
    model.eval()

    # Standard ImageNet normalization
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess, device


def segment_head_yolo_instances(
    image: np.ndarray, weights: Path, imgsz: int
) -> List[np.ndarray]:
    model = load_yolo_model(str(weights))
    result = model.predict(source=image, imgsz=imgsz, verbose=False)[0]

    masks: List[np.ndarray] = []
    if result.masks is not None and result.masks.data is not None:
        for m in result.masks.data.cpu().numpy():
            masks.append(m)

    if not masks and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 1
            masks.append(mask)

    if not masks:
        masks = [np.ones(image.shape[:2], dtype=np.uint8)]

    normalized = []
    for mask in masks:
        if mask.shape[:2] != image.shape[:2]:
            mask = np.array(
                Image.fromarray(to_uint8_mask(mask) * 255).resize(
                    (image.shape[1], image.shape[0]), resample=Image.NEAREST
                )
            )
        normalized.append(to_uint8_mask(mask))

    return normalized


def segment_head_yolo(image: np.ndarray, weights: Path, imgsz: int) -> Tuple[np.ndarray, np.ndarray]:
    masks = segment_head_yolo_instances(image, weights, imgsz)
    areas = [m.sum() for m in masks]
    idx = int(np.argmax(areas))
    return image, masks[idx]


def segment_head_mock(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.ones(image.shape[:2], dtype=np.uint8)
    return image, mask


def crop_with_mask(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = to_uint8_mask(mask)
    bbox = mask_bbox(mask)
    if bbox is None:
        return image, mask
    x1, y1, x2, y2 = bbox
    x2 = min(x2 + 1, image.shape[1])
    y2 = min(y2 + 1, image.shape[0])
    cropped = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    return cropped, cropped_mask


def resnet_embedding(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    model, preprocess, device = load_resnet_model()

    # Apply mask to image (black background)
    masked_img = image.copy()
    if mask is not None:
        mask_uint8 = to_uint8_mask(mask)
        masked_img[mask_uint8 == 0] = 0

    # Convert to PIL and preprocess
    pil_img = Image.fromarray(masked_img)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        features = model(input_tensor)

    # Flatten and normalize
    vec = features.cpu().numpy().flatten()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_centroids(embeddings: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    centroids = {}
    for name, vecs in embeddings.items():
        if not vecs:
            continue
        centroid = np.mean(np.stack(vecs, axis=0), axis=0)
        norm = np.linalg.norm(centroid)
        centroids[name] = centroid / norm if norm > 0 else centroid
    return centroids


def classify_query(
    image_path: Path,
    centroids: Dict[str, np.ndarray],
    backend: str,
    weights: Path,
    imgsz: int,
    topk: int,
) -> List[Dict[str, object]]:
    image = read_rgb(image_path)
    if backend == "yolo":
        masks = segment_head_yolo_instances(image, weights, imgsz)
    else:
        masks = [segment_head_mock(image)[1]]

    targets: List[Dict[str, object]] = []
    for mask in masks:
        cropped, cropped_mask = crop_with_mask(image, mask)

        # Use ResNet features
        emb = resnet_embedding(cropped, cropped_mask)

        scores = []
        for name, centroid in centroids.items():
            score = float(np.dot(emb, centroid))
            scores.append({"name": name, "score": score})
        scores.sort(key=lambda x: x["score"], reverse=True)

        targets.append(
            {
                "bbox": mask_bbox(mask),
                "predictions": scores[: max(topk, 1)],
                "_mask": mask,
            }
        )

    return targets


class FurryClassifier:
    def __init__(self, lib_images: List[Tuple[str, Path]], backend="yolo", weights="furry-03_best.pt", imgsz=640):
        self.backend = backend
        self.weights = Path(weights)
        self.imgsz = imgsz
        self.centroids = {}
        self.build_index(lib_images)

    def build_index(self, lib_images: List[Tuple[str, Path]]):
        embeddings: Dict[str, List[np.ndarray]] = {}
        print(f"Building index with {len(lib_images)} images...")
        for label, img_path in lib_images:
            if not img_path.exists():
                continue

            try:
                image = read_rgb(img_path)
                if self.backend == "yolo":
                    image, mask = segment_head_yolo(image, self.weights, self.imgsz)
                else:
                    image, mask = segment_head_mock(image)

                cropped, cropped_mask = crop_with_mask(image, mask)
                emb = resnet_embedding(cropped, cropped_mask)
                embeddings.setdefault(label, []).append(emb)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        self.centroids = compute_centroids(embeddings)
        print(f"Index built with {len(self.centroids)} classes.")

    def predict(self, image_path: Path, topk: int = 1) -> List[Dict[str, object]]:
        if not self.centroids:
            return []
        return classify_query(image_path, self.centroids, self.backend, self.weights, self.imgsz, topk)

