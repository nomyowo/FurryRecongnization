import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# 特征版本号：当预处理 / 提特征逻辑发生变化时递增，使旧缓存自动失效。
FEATURE_VERSION = 2
# 默认特征 backbone，可用环境变量覆盖：resnet18 | dinov2
DEFAULT_FEATURE_BACKBONE = os.environ.get("FURRY_BACKBONE", "resnet18")
# 批处理大小（提特征时一次喂给 GPU/CPU 的图片数）
EMBED_BATCH_SIZE = int(os.environ.get("FURRY_EMBED_BATCH", "64"))


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


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """将图片用 0（黑色）填充为正方形，避免后续 Resize 拉伸破坏长宽比。"""
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    if image.ndim == 3:
        canvas = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((size, size), dtype=image.dtype)
    top = (size - h) // 2
    left = (size - w) // 2
    canvas[top:top + h, left:left + w] = image
    return canvas


@lru_cache(maxsize=1)
def load_yolo_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


# ---------------------------------------------------------------------------
# 特征提取 backbone：返回 (model, preprocess, device, dim, name)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def load_feature_model(backbone: str = "resnet18"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()  # 取 avgpool 后的 512 维特征
        dim = 512
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm,
        ])
    elif backbone == "dinov2":
        # 自监督预训练，特征判别力优于 ImageNet ResNet18；首次加载需联网下权重。
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        dim = 768
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm,
        ])
    else:
        raise ValueError(f"Unknown feature backbone: {backbone}")

    model.to(device)
    model.eval()
    return model, preprocess, device, dim, backbone


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


def _prepare_masked_image(image: np.ndarray, mask: Optional[np.ndarray]) -> Image.Image:
    """应用掩码（背景置黑）、pad 成正方形，返回 PIL 图。"""
    masked_img = image.copy()
    if mask is not None:
        mask_uint8 = to_uint8_mask(mask)
        masked_img[mask_uint8 == 0] = 0
    masked_img = pad_to_square(masked_img)
    return Image.fromarray(masked_img)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """对每行做 L2 归一化（支持一维向量或二维矩阵）。"""
    if mat.ndim == 1:
        norm = np.linalg.norm(mat)
        return mat / norm if norm > 0 else mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def resnet_embedding(image: np.ndarray, mask: np.ndarray, backbone: str = "resnet18") -> np.ndarray:
    """单图提特征（保留旧接口）。批量场景请用 embed_batch。"""
    return embed_batch([(image, mask)], backbone)[0]


def embed_batch(
    items: List[Tuple[np.ndarray, Optional[np.ndarray]]],
    backbone: str = "resnet18",
) -> np.ndarray:
    """批量提取并 L2 归一化特征。

    items: [(image, mask), ...]
    返回: shape [len(items), dim] 的归一化向量矩阵。
    """
    if not items:
        return np.zeros((0, 0), dtype=np.float32)

    model, preprocess, device, dim, _ = load_feature_model(backbone)

    out = np.zeros((len(items), dim), dtype=np.float32)
    for start in range(0, len(items), EMBED_BATCH_SIZE):
        chunk = items[start:start + EMBED_BATCH_SIZE]
        tensors = [preprocess(_prepare_masked_image(img, m)) for img, m in chunk]
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            feats = model(batch)
        out[start:start + len(chunk)] = feats.cpu().numpy().astype(np.float32)

    return _l2_normalize(out)


# ---------------------------------------------------------------------------
# 特征缓存：按文件名缓存库图特征，避免每次增删图全量重算
# ---------------------------------------------------------------------------

class EmbeddingCache:
    def __init__(self, backbone: str, cache_dir: str = "."):
        self.backbone = backbone
        self.path = os.path.join(cache_dir, f"embeddings_cache_{backbone}.pkl")
        # filename -> {"label": str, "vec": np.ndarray, "mtime": float}
        self.entries: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            if data.get("version") == FEATURE_VERSION and data.get("backbone") == self.backbone:
                self.entries = data.get("entries", {})
            else:
                print(f"Embedding cache outdated, ignoring {self.path}")
        except Exception as e:
            print(f"Failed to load embedding cache: {e}")

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(
                    {"version": FEATURE_VERSION, "backbone": self.backbone, "entries": self.entries},
                    f,
                )
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"Failed to save embedding cache: {e}")

    def get(self, filename: str, mtime: float) -> Optional[np.ndarray]:
        e = self.entries.get(filename)
        if e is not None and e.get("mtime") == mtime:
            return e["vec"]
        return None

    def put(self, filename: str, label: str, vec: np.ndarray, mtime: float):
        self.entries[filename] = {"label": label, "vec": vec, "mtime": mtime}

    def update_label(self, filename: str, label: str):
        if filename in self.entries:
            self.entries[filename]["label"] = label

    def prune(self, valid_filenames: set):
        for fn in list(self.entries.keys()):
            if fn not in valid_filenames:
                del self.entries[fn]


def compute_centroids(embeddings: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    centroids = {}
    for name, vecs in embeddings.items():
        if not vecs:
            continue
        centroid = np.mean(np.stack(vecs, axis=0), axis=0)
        norm = np.linalg.norm(centroid)
        centroids[name] = centroid / norm if norm > 0 else centroid
    return centroids


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------

class FurryClassifier:
    def __init__(
        self,
        lib_images: List[Tuple[str, Path]],
        backend: str = "yolo",
        weights: str = "furry-03_best.pt",
        imgsz: int = 640,
        feature_backbone: str = DEFAULT_FEATURE_BACKBONE,
        knn_k: int = 5,
        score_mode: str = "knn",  # "knn"（每类 top-k 相似度均值）或 "centroid"
    ):
        self.backend = backend
        self.weights = Path(weights)
        self.imgsz = imgsz
        self.feature_backbone = feature_backbone
        self.knn_k = knn_k
        self.score_mode = score_mode

        # 画廊矩阵：所有库图向量堆叠，用于向量化匹配
        self.gallery: np.ndarray = np.zeros((0, 0), dtype=np.float32)  # [N, D]
        self.gallery_labels: List[str] = []                            # 长度 N
        self.label_names: List[str] = []                               # 去重后的类名
        # 兼容旧字段 / centroid 模式
        self.centroids: Dict[str, np.ndarray] = {}

        self.build_index(lib_images)

    def build_index(self, lib_images: List[Tuple[str, Path]]):
        cache = EmbeddingCache(self.feature_backbone)
        print(f"Building index with {len(lib_images)} images (backbone={self.feature_backbone})...")

        valid_filenames = set()
        to_compute: List[Tuple[str, str, np.ndarray, np.ndarray, float]] = []  # (fn,label,img,mask,mtime)
        cached_vecs: Dict[str, Tuple[str, np.ndarray]] = {}  # fn -> (label, vec)

        for label, img_path in lib_images:
            if not img_path.exists():
                continue
            filename = img_path.name
            valid_filenames.add(filename)
            try:
                mtime = os.path.getmtime(img_path)
            except OSError:
                mtime = 0.0

            cached = cache.get(filename, mtime)
            if cached is not None:
                cache.update_label(filename, label)  # 标签可能变了，向量不变
                cached_vecs[filename] = (label, cached)
                continue

            # 需要重新计算：先做分割 + 裁剪
            try:
                image = read_rgb(img_path)
                if self.backend == "yolo":
                    image, mask = segment_head_yolo(image, self.weights, self.imgsz)
                else:
                    image, mask = segment_head_mock(image)
                cropped, cropped_mask = crop_with_mask(image, mask)
                to_compute.append((filename, label, cropped, cropped_mask, mtime))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # 批量提特征（仅针对新增/变更的图）
        if to_compute:
            print(f"  computing {len(to_compute)} new embeddings, reusing {len(cached_vecs)} cached")
            vecs = embed_batch([(c[2], c[3]) for c in to_compute], self.feature_backbone)
            for (filename, label, _, _, mtime), vec in zip(to_compute, vecs):
                cache.put(filename, label, vec, mtime)
                cached_vecs[filename] = (label, vec)
        else:
            print(f"  all {len(cached_vecs)} embeddings served from cache")

        cache.prune(valid_filenames)
        cache.save()

        # 组装画廊矩阵
        gallery_vecs: List[np.ndarray] = []
        gallery_labels: List[str] = []
        for filename, (label, vec) in cached_vecs.items():
            gallery_vecs.append(vec)
            gallery_labels.append(label)

        if gallery_vecs:
            self.gallery = np.stack(gallery_vecs, axis=0).astype(np.float32)
            self.gallery_labels = gallery_labels
            self.label_names = sorted(set(gallery_labels))
        else:
            self.gallery = np.zeros((0, 0), dtype=np.float32)
            self.gallery_labels = []
            self.label_names = []

        # 同时维护质心（供 centroid 模式 / 兼容旧接口）
        emb_by_label: Dict[str, List[np.ndarray]] = {}
        for vec, label in zip(gallery_vecs, gallery_labels):
            emb_by_label.setdefault(label, []).append(vec)
        self.centroids = compute_centroids(emb_by_label)

        print(f"Index built: {len(self.gallery_labels)} vectors, {len(self.label_names)} classes.")

    def _score_labels(self, emb: np.ndarray) -> List[Dict[str, object]]:
        """对单个查询向量，向量化地给每个类打分并降序返回。"""
        if self.score_mode == "centroid" or self.gallery.size == 0:
            scores = [
                {"name": name, "score": float(np.dot(emb, c))}
                for name, c in self.centroids.items()
            ]
            scores.sort(key=lambda x: x["score"], reverse=True)
            return scores

        # k-NN：一次矩阵乘得到与所有库向量的相似度
        sims = self.gallery @ emb  # [N]
        labels = np.asarray(self.gallery_labels)

        results = []
        for name in self.label_names:
            label_sims = sims[labels == name]
            if label_sims.size == 0:
                continue
            k = min(self.knn_k, label_sims.size)
            topk = np.partition(label_sims, -k)[-k:]
            results.append({"name": name, "score": float(topk.mean())})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def predict(self, image_path: Path, topk: int = 1) -> List[Dict[str, object]]:
        if self.gallery.size == 0 and not self.centroids:
            return []

        image = read_rgb(image_path)
        if self.backend == "yolo":
            masks = segment_head_yolo_instances(image, self.weights, self.imgsz)
        else:
            masks = [segment_head_mock(image)[1]]

        # 先批量提取所有检测区域的特征
        crops = [crop_with_mask(image, m) for m in masks]
        embs = embed_batch([(c, cm) for c, cm in crops], self.feature_backbone)

        targets: List[Dict[str, object]] = []
        for mask, emb in zip(masks, embs):
            scores = self._score_labels(emb)
            targets.append(
                {
                    "bbox": mask_bbox(mask),
                    "predictions": scores[: max(topk, 1)],
                    "_mask": mask,
                }
            )
        return targets
