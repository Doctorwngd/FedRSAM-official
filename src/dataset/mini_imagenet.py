from __future__ import print_function

import csv
import os

import numpy as np
import torch.utils.data as data
from PIL import Image
from tqdm.autonotebook import tqdm

__all__ = ['MiniImageNetDataset']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.webp'}


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import torchvision.datasets.accimage as accimage

    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    return pil_loader(path)


class MiniImageNetPaths:
    ROOT_CANDIDATES = (
        '',
        'mini-imagenet',
        'mini_imagenet',
        'miniimagenet',
        'miniImageNet',
    )
    SPLIT_ALIASES = {
        'train': ('train',),
        'val': ('val', 'valid', 'validation', 'test'),
        'test': ('test', 'val', 'valid', 'validation'),
    }
    CSV_FILENAMES = {
        'train': ('train.csv',),
        'val': ('val.csv', 'valid.csv', 'validation.csv'),
        'test': ('test.csv', 'val.csv', 'valid.csv', 'validation.csv'),
    }

    def __init__(self, root_dir):
        self.root_dir = self._resolve_root(root_dir)
        self.layout = self._infer_layout()
        self.paths = {
            'train': self._make_paths('train'),
            'val': self._make_paths('val'),
            'test': self._make_paths('test'),
        }

    def _resolve_root(self, root_dir):
        candidates = []
        if self._looks_like_root(root_dir):
            candidates.append(root_dir)

        parent = os.path.dirname(root_dir)
        for candidate in self.ROOT_CANDIDATES:
            candidate_root = root_dir if candidate == '' else os.path.join(parent, candidate)
            if self._looks_like_root(candidate_root):
                candidates.append(candidate_root)

        if candidates:
            return candidates[0]

        raise FileNotFoundError(
            'Mini-ImageNet root not found: {}. Tried common variants under {}.'.format(root_dir, parent)
        )

    def _looks_like_root(self, root_dir):
        if not os.path.isdir(root_dir):
            return False

        for split in ('train', 'val', 'test'):
            if self._find_csv_in_root(root_dir, split) is not None:
                return True

        for split in ('train', 'val', 'test'):
            if self._find_split_dir_in_root(root_dir, split) is not None:
                return True

        return bool(self._list_class_dirs(root_dir))

    def _infer_layout(self):
        for split in ('train', 'val', 'test'):
            if self._find_csv(split) is not None:
                return 'csv'

        for split in ('train', 'val', 'test'):
            if self._find_split_dir(split) is not None:
                return 'folder'

        if self._list_class_dirs(self.root_dir):
            return 'flat-folder'

        raise FileNotFoundError(
            'Unsupported Mini-ImageNet layout under {}. Expected split folders or split CSV files.'.format(
                self.root_dir
            )
        )

    def _find_csv(self, split):
        return self._find_csv_in_root(self.root_dir, split)

    def _find_csv_in_root(self, root_dir, split):
        for filename in self.CSV_FILENAMES[split]:
            csv_path = os.path.join(root_dir, filename)
            if os.path.isfile(csv_path):
                return csv_path
        return None

    def _find_split_dir(self, split):
        return self._find_split_dir_in_root(self.root_dir, split)

    def _find_split_dir_in_root(self, root_dir, split):
        for alias in self.SPLIT_ALIASES[split]:
            split_dir = os.path.join(root_dir, alias)
            if os.path.isdir(split_dir) and self._list_class_dirs(split_dir):
                return split_dir
        return None

    def _collect_image_files(self, root_dir):
        image_paths = []
        for current_root, _, filenames in os.walk(root_dir):
            for filename in sorted(filenames):
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(current_root, filename))
        return sorted(image_paths)

    def _dir_has_images(self, root_dir):
        return len(self._collect_image_files(root_dir)) > 0

    def _list_class_dirs(self, root_dir):
        class_dirs = []
        for name in sorted(os.listdir(root_dir)):
            full_path = os.path.join(root_dir, name)
            if os.path.isdir(full_path) and not name.startswith('.') and self._dir_has_images(full_path):
                class_dirs.append(name)
        return class_dirs

    def _read_csv_rows(self, csv_path):
        with open(csv_path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise RuntimeError('CSV {} has no header row'.format(csv_path))

            filename_key = None
            label_key = None
            for candidate in ('filename', 'file', 'filepath', 'path', 'image', 'image_name'):
                if candidate in reader.fieldnames:
                    filename_key = candidate
                    break
            for candidate in ('label', 'class', 'label_name', 'wnid'):
                if candidate in reader.fieldnames:
                    label_key = candidate
                    break

            if filename_key is None or label_key is None:
                raise RuntimeError(
                    'CSV {} must contain filename/path and label columns; got {}'.format(
                        csv_path, reader.fieldnames
                    )
                )

            rows = []
            for row in reader:
                rows.append((row[filename_key], row[label_key]))
            return rows

    def _resolve_image_path(self, image_name):
        candidates = []
        if os.path.isabs(image_name):
            candidates.append(image_name)
        else:
            candidates.append(os.path.join(self.root_dir, image_name))
            candidates.append(os.path.join(self.root_dir, 'images', image_name))
            candidates.append(os.path.join(self.root_dir, 'all_images', image_name))

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError('Mini-ImageNet image not found: {}'.format(image_name))

    def _make_paths(self, split):
        if self.layout == 'csv':
            csv_path = self._find_csv(split)
            if csv_path is None and split == 'test':
                csv_path = self._find_csv('val')
            if csv_path is None and split == 'val':
                csv_path = self._find_csv('test')
            if csv_path is None:
                return []
            rows = self._read_csv_rows(csv_path)
            return [
                (self._resolve_image_path(image_name), label)
                for image_name, label in rows
            ]

        if self.layout == 'folder':
            split_dir = self._find_split_dir(split)
            if split_dir is None and split == 'test':
                split_dir = self._find_split_dir('val')
            if split_dir is None and split == 'val':
                split_dir = self._find_split_dir('test')
            if split_dir is None:
                return []
            samples = []
            for label in sorted(self._list_class_dirs(split_dir)):
                class_dir = os.path.join(split_dir, label)
                for image_path in self._collect_image_files(class_dir):
                    samples.append((image_path, label))
            return samples

        samples = []
        for label in sorted(self._list_class_dirs(self.root_dir)):
            class_dir = os.path.join(self.root_dir, label)
            for image_path in self._collect_image_files(class_dir):
                samples.append((image_path, label))
        return samples


class MiniImageNetDataset(data.Dataset):
    def __init__(
        self,
        root,
        split='train',
        preload=False,
        load_transform=None,
        transform=None,
        download=False,
        max_samples=None,
    ):
        if download:
            raise NotImplementedError('Mini-ImageNet download is not implemented.')

        mip = MiniImageNetPaths(root)
        self.split = split
        self.preload = preload
        self.transform = transform
        self.transform_results = {}
        self.loader = default_loader
        self.max_samples = max_samples

        normalized_split = 'val' if split == 'validation' else split
        if normalized_split == 'test':
            source_split = 'val' if len(mip.paths.get('val', [])) > 0 else 'test'
        else:
            source_split = 'train' if normalized_split == 'train' else normalized_split

        if source_split not in mip.paths:
            raise ValueError('Unsupported split: {}'.format(split))

        raw_samples = list(mip.paths[source_split])
        if len(raw_samples) == 0:
            raise RuntimeError('No Mini-ImageNet samples found for split {} under {}'.format(source_split, mip.root_dir))

        if self.max_samples is not None:
            sample_count = min(self.max_samples, len(raw_samples))
            sample_indices = np.random.permutation(len(raw_samples))[:sample_count]
            raw_samples = [raw_samples[index] for index in sample_indices]

        raw_labels = [sample[1] for sample in raw_samples]
        self.classes = sorted(set(raw_labels))
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        self.samples = [(image_path, self.class_to_idx[label]) for image_path, label in raw_samples]

        self.data = np.array([sample[0] for sample in self.samples])
        self.targets = np.array([sample[1] for sample in self.samples], dtype=np.int64)
        print('[MiniImageNetDataset] split={} source_split={} samples={} classes={} min_target={} max_target={}'.format(
            split,
            source_split,
            len(self.samples),
            len(self.classes),
            int(self.targets.min()) if len(self.targets) > 0 else -1,
            int(self.targets.max()) if len(self.targets) > 0 else -1,
        ))

        self.img_data = []
        self.label_data = []
        if self.preload:
            for sample in tqdm(self.samples, desc='Preloading {} data...'.format(split)):
                self.img_data.append(self.loader(sample[0]))
                self.label_data.append(sample[1])

            if load_transform:
                transform_fns = load_transform
                if not isinstance(transform_fns, (list, tuple)):
                    transform_fns = [transform_fns]
                for transform_fn in transform_fns:
                    load_result = transform_fn(self.img_data, self.label_data)
                    self.img_data, self.label_data = load_result[:2]
                    if len(load_result) > 2:
                        self.transform_results.update(load_result[2])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            target = self.label_data[idx]
        else:
            img, target = self.data[idx], self.targets[idx]
            img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target
