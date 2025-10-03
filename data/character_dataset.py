import json
import os
from typing import Callable, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class UnicodeCharacterDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_root: str,
        split_file: str,
        label_map_file: str,
        transform: Callable | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.data_root = data_root
        self.transform = transform

        self.label_map = self._load_label_map(label_map_file)
        samples = self._load_split(split_file, split)

        self.samples: List[Tuple[str, int]] = []
        self.targets: List[int] = []

        for entry in samples:
            unicode_id = entry["unicode"]
            if unicode_id not in self.label_map:
                continue
            label = self.label_map[unicode_id]
            rel_path = entry["relative_path"]
            abs_path = os.path.join(self.data_root, rel_path)
            self.samples.append((abs_path, label))
            self.targets.append(label)

        if not self.samples:
            raise RuntimeError(f"No samples found for split '{split}' using split_file='{split_file}'.")

        self.nb_classes = len(self.label_map)

    @staticmethod
    def _load_label_map(label_map_file: str) -> Dict[str, int]:
        if not os.path.isfile(label_map_file):
            raise FileNotFoundError(f"Label map file not found: {label_map_file}")
        with open(label_map_file, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        return {str(k): int(v) for k, v in label_map.items()}

    @staticmethod
    def _load_split(split_file: str, split: str) -> List[Dict[str, str]]:
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split metadata file not found: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if split not in data:
            raise KeyError(f"Split '{split}' not found in {split_file}")
        return data[split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target
