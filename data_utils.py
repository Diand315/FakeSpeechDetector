import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

__author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta: str, is_train: bool = False, is_eval: bool = False):
    meta_dict = {}
    file_list = []
    with open(dir_meta, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if is_train:
        for line in lines:
            parts = line.strip().split(" ")
            _, key, _, system_id, label = parts
            file_list.append(key)
            meta_dict[key] = {"label": 1 if label == "bonafide" else 0, "system": system_id}
        return meta_dict, file_list
    elif is_eval:
        for line in lines:
            parts = line.strip().split(" ")
            _, key, _, _, _ = parts
            file_list.append(key)
        return file_list
    else:
        for line in lines:
            parts = line.strip().split(" ")
            _, key, _, system_id, label = parts
            file_list.append(key)
            meta_dict[key] = {"label": 1 if label == "bonafide" else 0, "system": system_id}
        return meta_dict, file_list


def pad(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        start = np.random.randint(0, x_len - max_len + 1)
        return x[start:start + max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def add_random_noise(audio: np.ndarray, noise_factor: float = 0.003):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, ext="flac", subfolder="flac", auto_subfolder=False):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.ext = ext
        self.subfolder = subfolder
        self.auto_subfolder = auto_subfolder
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        if self.auto_subfolder:
            if "CFAD" in str(self.base_dir):
                if key.startswith("LA_D_"):
                    first_folder = "dev_clean"
                elif key.startswith("LA_T_"):
                    first_folder = "train_clean"
                else:
                    first_folder = self.subfolder
                info = self.labels.get(key)
                if info is not None and isinstance(info, dict):
                    system_id = info["system"].upper()
                    label = info["label"]
                    if system_id.startswith("R") or label == 1:
                        second_folder = "real_clean"
                    elif system_id.startswith("A") or label == 0:
                        second_folder = "fake_clean"
                    else:
                        second_folder = self.subfolder
                    folder = f"{first_folder}/{second_folder}"
                else:
                    folder = first_folder
            else:
                if key.startswith("LA_D_"):
                    folder = "dev_clean"
                elif key.startswith("LA_T_"):
                    folder = "train_clean"
                else:
                    folder = self.subfolder
        else:
            folder = self.subfolder

        file_path = self.base_dir / folder / f"{key}.{self.ext}"
        audio, _ = sf.read(str(file_path))
        if "CFAD" in str(self.base_dir):
            audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
            if np.random.rand() < 0.5:
                audio = add_random_noise(audio, noise_factor=np.random.uniform(0.001, 0.005))
        audio_pad = pad_random(audio, self.cut)
        x_input = torch.Tensor(audio_pad)
        y = self.labels.get(key, {"label": 0})["label"] if isinstance(self.labels.get(key), dict) else self.labels.get(key)
        return x_input, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, ext="flac", subfolder="flac", auto_subfolder=False, labels=None):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.ext = ext
        self.subfolder = subfolder
        self.auto_subfolder = auto_subfolder
        self.labels = labels
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        if self.auto_subfolder:
            if "CFAD" in str(self.base_dir) and self.labels is not None:
                if key.startswith("LA_D_"):
                    first_folder = "dev_clean"
                elif key.startswith("LA_T_"):
                    first_folder = "train_clean"
                else:
                    first_folder = self.subfolder
                info = self.labels.get(key)
                if info is not None and isinstance(info, dict):
                    system_id = info["system"].upper()
                    label = info["label"]
                    if system_id.startswith("R") or label == 1:
                        second_folder = "real_clean"
                    elif system_id.startswith("A") or label == 0:
                        second_folder = "fake_clean"
                    else:
                        second_folder = self.subfolder
                    folder = f"{first_folder}/{second_folder}"
                else:
                    folder = first_folder
            else:
                if key.startswith("LA_D_"):
                    folder = "dev_clean"
                elif key.startswith("LA_T_"):
                    folder = "train_clean"
                else:
                    folder = self.subfolder
        else:
            folder = self.subfolder

        file_path = self.base_dir / folder / f"{key}.{self.ext}"
        audio, _ = sf.read(str(file_path))
        if "CFAD" in str(self.base_dir):
            audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
        audio_pad = pad(audio, self.cut)
        x_input = torch.Tensor(audio_pad)
        return x_input, key