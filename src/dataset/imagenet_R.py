from __future__ import print_function

import contextlib
import os
from collections import defaultdict

import imageio
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm.autonotebook import tqdm

from dataset.base import *

__all__ = ['ImageNetRDataset']


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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
    else:
        return pil_loader(path)


def build_set(root, split, imgs, noise_type='pairflip', noise_rate=0.5):
    tmp_imgs = imgs
    _ = np.argsort(tmp_imgs)


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    while img.shape[-1] < total_channels:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


def _stratified_split(samples, test_ratio=0.2, seed=0):
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    train_samples = []
    test_samples = []
    rng = np.random.RandomState(seed)

    for class_idx in sorted(by_class.keys()):
        class_samples = by_class[class_idx]
        indices = np.arange(len(class_samples))
        rng.shuffle(indices)

        n_test = max(1, int(round(len(class_samples) * test_ratio)))
        if len(class_samples) > 1:
            n_test = min(n_test, len(class_samples) - 1)
        else:
            n_test = 1

        test_indices = set(indices[:n_test].tolist())
        for i, sample in enumerate(class_samples):
            if i in test_indices:
                test_samples.append(sample)
            else:
                train_samples.append(sample)

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


class ImageNetRDataset(data.Dataset):
    def __init__(self, root, split='train', preload=False, load_transform=None,
                 transform=None, download=False, max_samples=None,
                 test_ratio=0.2, seed=0):
        if download:
            download_and_unzip('', root)

        if split not in ['train', 'test']:
            raise ValueError("split must be 'train' or 'test', got {}".format(split))

        self.root = root
        self.split = split
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()
        self.loader = default_loader
        self.max_samples = max_samples
        self.test_ratio = test_ratio
        self.seed = seed

        base_dataset = datasets.ImageFolder(root=root, loader=self.loader)
        self.class_to_idx = dict(base_dataset.class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(range(len(base_dataset.classes)))

        all_samples = list(base_dataset.samples)
        train_samples, test_samples = _stratified_split(
            all_samples, test_ratio=self.test_ratio, seed=self.seed
        )

        self.samples = train_samples if split == 'train' else test_samples
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            perm = np.random.permutation(self.samples_num)
            self.samples = [self.samples[i] for i in perm]

        self.data = np.array([s[0] for s in self.samples])
        self.targets = np.array([s[1] for s in self.samples], dtype=np.int64)

        self.IMAGE_SHAPE = None
        self.img_data = []
        self.label_data = []

        if self.preload:
            load_desc = "Preloading {} data...".format(split)
            first_img = imageio.imread(self.samples[0][0])
            first_img = _add_channels(first_img)
            self.IMAGE_SHAPE = first_img.shape

            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=np.int64)

            for idx in tqdm(range(self.samples_num), desc=load_desc):
                img_path, target = self.samples[idx]
                img = imageio.imread(img_path)
                img = _add_channels(img)
                self.img_data[idx] = img
                self.label_data[idx] = target

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            target = int(self.label_data[idx])
            img = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
        else:
            img_path, target = self.data[idx], int(self.targets[idx])
            img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


def tiny_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def tiny_noniid(dataset, num_users):
    class_per_user = 1
    num_shards = num_users * class_per_user
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)

    labels = []
    for element in dataset:
        labels.append(int(element[1]))
    labels = np.array(labels)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, class_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = set(np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0))
    return dict_users



def tiny_dirichlet_unbalanced(dataset, n_nets, alpha=0.5):
    y_train = torch.zeros(len(dataset), dtype=torch.long)
    print(y_train.dtype)
    for a in range(len(dataset)):
        y_train[a] = dataset[a][1]

    min_size = 0
    K = len(dataset.class_to_idx)
    N = y_train.shape[0]
    net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_nets)}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            proportions = np.array([
                p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)
            ])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def tiny_dirichlet_balanced(dataset, n_nets, alpha=0.5):
    with temp_seed(0):
        y_train = torch.zeros(len(dataset), dtype=torch.long)

        for a in range(len(dataset)):
            y_train[a] = dataset[a][1]

        K = len(dataset.class_to_idx)
        N = y_train.shape[0]
        print(N)
        net_dataidx_map = {i: np.array([], dtype='int64') for i in range(n_nets)}
        assigned_ids = []
        idx_batch = [[] for _ in range(n_nets)]
        num_data_per_client = int(N / n_nets)

        for i in range(n_nets):
            weights = torch.zeros(N)
            proportions = np.random.dirichlet(np.repeat(alpha, K))
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                weights[idx_k] = proportions[k]
            weights[assigned_ids] = 0.0
            idx_batch[i] = torch.multinomial(
                weights, num_data_per_client, replacement=False
            ).tolist()
            assigned_ids += idx_batch[i]

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    exs = ImageNetRDataset('../data/imagenet-r', split='train', transform=transform)

    mean = 0
    sq_mean = 0
    for ex in exs:
        mean += ex[0].sum(1).sum(1) / (224 * 224)
        sq_mean += ex[0].pow(2).sum(1).sum(1) / (224 * 224)

    mean /= len(exs)
    sq_mean /= len(exs)

    std = (sq_mean - mean.pow(2)).pow(0.5)

    print('train size:', len(exs))
    print(mean)
    print(std)