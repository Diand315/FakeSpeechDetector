import argparse
import json
import os
import re
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2019_devNeval, genSpoof_list
)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)

# Enable anomaly detection for debugging (detects NaN/Infs in backward)
torch.autograd.set_detect_anomaly(True)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, Freq_aug=False):
        logpt = F.log_softmax(x, dim=1)
        pt = torch.exp(logpt)
        logpt = ((1 - pt) ** self.gamma) * logpt
        loss = F.nll_loss(logpt, torch.argmax(x, dim=1), weight=self.weight, reduction=self.reduction)
        return loss


def convert_label(label):
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        label_lower = label.lower().strip()
        if label_lower in ["0", "1"]:
            return int(label_lower)
        if "bonafide" in label_lower:
            return 1
        return 0
    raise ValueError(f"Label of unsupported type: {type(label)}")


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        else:
            progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def spec_augment(spectrogram: torch.Tensor, time_mask_param: int = 30, freq_mask_param: int = 13) -> torch.Tensor:
    batch_size, num_freq, num_time = spectrogram.shape
    t = random.randint(0, time_mask_param)
    if num_time - t > 0:
        t0 = random.randint(0, num_time - t)
        spectrogram[:, :, t0:t0+t] = 0
    f = random.randint(0, freq_mask_param)
    if num_freq - f > 0:
        f0 = random.randint(0, num_freq - f)
        spectrogram[:, f0:f0+f, :] = 0
    return spectrogram


def extra_data_augment(waveform: torch.Tensor, config: dict) -> torch.Tensor:
    if config.get("use_spec_augment", False):
        n_fft = config.get("n_fft", 512)
        hop_length = config.get("hop_length", 256)
        win_length = config.get("win_length", 512)
        time_mask_param = config.get("time_mask_param", 30)
        freq_mask_param = config.get("freq_mask_param", 13)
        spectrogram = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length, return_complex=True)
        spectrogram = torch.abs(spectrogram)
        return spec_augment(spectrogram, time_mask_param, freq_mask_param)
    else:
        return waveform


def combine_loaders(loader_list: List[DataLoader], config: dict, seed: int,
                    shuffle: bool, drop_last: bool) -> DataLoader:
    from torch.utils.data import Dataset
    if not loader_list:
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0
            def __getitem__(self, index):
                raise IndexError("Empty dataset")
        empty_dataset = EmptyDataset()
        return DataLoader(empty_dataset, batch_size=config["batch_size"],
                          shuffle=False, drop_last=drop_last, pin_memory=True,
                          worker_init_fn=seed_worker)
    datasets = [loader.dataset for loader in loader_list if hasattr(loader, "dataset")]
    if not datasets:
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0
            def __getitem__(self, index):
                raise IndexError("Empty dataset")
        empty_dataset = EmptyDataset()
        return DataLoader(empty_dataset, batch_size=config["batch_size"],
                          shuffle=False, drop_last=drop_last, pin_memory=True,
                          worker_init_fn=seed_worker)
    combined_dataset = ConcatDataset(datasets)
    if len(combined_dataset) == 0:
        shuffle = False
    return DataLoader(combined_dataset, batch_size=config["batch_size"],
                      shuffle=shuffle, drop_last=drop_last, pin_memory=True,
                      worker_init_fn=seed_worker)


def get_loader(la_database_path: Path, seed: int, config: dict) -> List[DataLoader]:
    trn_loaders = []
    dev_loaders = []
    eval_loaders = []
    if config.get("run_la", False):
        la_trn_loader, la_dev_loader, la_eval_loader = get_loader_la(la_database_path, seed, config)
        trn_loaders.append(la_trn_loader)
        dev_loaders.append(la_dev_loader)
        eval_loaders.append(la_eval_loader)
    run_cfad = config.get("run_cfad", [])
    if isinstance(run_cfad, str):
        run_cfad = [run_cfad]
    cfad_root = Path(config["cfad_path"])
    if "train" in run_cfad:
        cfad_trn_loader = get_loader_cfad("cfad_train_protocol", "cfad_train_subfolder", config, cfad_root)
        trn_loaders.append(cfad_trn_loader)
    if "dev" in run_cfad:
        cfad_dev_loader = get_loader_cfad("cfad_dev_protocol", "cfad_dev_subfolder", config, cfad_root)
        dev_loaders.append(cfad_dev_loader)
    if "eval" in run_cfad:
        cfad_eval_loader = get_loader_cfad("cfad_eval_protocol", "cfad_eval_subfolder", config, cfad_root)
        eval_loaders.append(cfad_eval_loader)
    combined_trn_loader = combine_loaders(trn_loaders, config, seed, shuffle=True, drop_last=True)
    combined_dev_loader = combine_loaders(dev_loaders, config, seed, shuffle=False, drop_last=False)
    combined_eval_loader = combine_loaders(eval_loaders, config, seed, shuffle=False, drop_last=False)
    return combined_trn_loader, combined_dev_loader, combined_eval_loader


def get_loader_cfad(protocol_name: str, subfolder_name: str, config: dict, cfad_root: Path) -> DataLoader:
    from torch.utils.data import DataLoader
    cfad_audio_root = Path(config["cfad_audio_root"])
    dir_meta_name = Path(config[protocol_name])
    d_label, file_list = genSpoof_list(
        dir_meta=dir_meta_name,
        is_train=True,
        is_eval=(protocol_name == "cfad_eval_protocol")
    )
    print(f"CFAD no. files for {protocol_name}: {len(file_list)}")
    dataset = Dataset_ASVspoof2019_devNeval(
        list_IDs=file_list, labels=d_label, base_dir=cfad_audio_root,
        ext="wav", subfolder=config.get(subfolder_name), auto_subfolder=True)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                      drop_last=True, pin_memory=True, worker_init_fn=seed_worker)


def get_loader_la(la_database_path: Path, seed: int, config: dict) -> List[DataLoader]:
    track = config["track"]
    prefix = f"ASVspoof2019.{track}"
    trn_database_path = la_database_path / f"ASVspoof2019_{track}_train"
    dev_database_path = la_database_path / f"ASVspoof2019_{track}_dev"
    eval_database_path = la_database_path / f"ASVspoof2019_{track}_eval"
    trn_list_path = la_database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix}.cm.train.trn.txt"
    dev_trial_path = la_database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix}.cm.dev.trl.txt"
    eval_trial_path = la_database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix}.cm.eval.trl.txt"
    
    d_label_trn, file_train = genSpoof_list(trn_list_path, is_train=True, is_eval=False)
    print("LA no. training files:", len(file_train))
    train_set = Dataset_ASVspoof2019_train(
        list_IDs=file_train, labels=d_label_trn, base_dir=trn_database_path,
        ext="wav", subfolder="", auto_subfolder=False)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set, batch_size=config["batch_size"],
                            shuffle=True, drop_last=True, pin_memory=True,
                            worker_init_fn=seed_worker, generator=gen)
    
    _, file_dev = genSpoof_list(dev_trial_path, is_train=False, is_eval=False)
    print("LA no. validation files:", len(file_dev))
    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev, base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set, batch_size=config["batch_size"],
                            shuffle=False, drop_last=False, pin_memory=True)
    
    file_eval = genSpoof_list(eval_trial_path, is_train=False, is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval, base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set, batch_size=config["batch_size"],
                             shuffle=False, drop_last=False, pin_memory=True)
    return trn_loader, dev_loader, eval_loader


def get_model(model_config: Dict, device: torch.device):
    module = __import__("models.AASIST", fromlist=["Model"])
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum(p.numel() for p in model.parameters())
    print("no. model params:{}".format(nb_params))
    return model


def produce_evaluation_file(data_loader: DataLoader, model, device: torch.device,
                            save_path: str, trial_path: str) -> None:
    model.eval()
    with open(trial_path, "r", encoding="utf-8") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            if torch.isnan(batch_out).any() or torch.isinf(batch_out).any():
                print("Warning: NaN or Inf detected in model output. Clamping values.")
                batch_out = torch.nan_to_num(batch_out, nan=0.0, posinf=1e6, neginf=-1e6)
            print("Batch Output stats - min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
                batch_out.min().item(), batch_out.max().item(), batch_out.mean().item()))
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
    print("Trial lines:", len(trial_lines))
    print("File names from loader:", len(fname_list))
    print("Score list length:", len(score_list))
    if not (len(trial_lines) == len(fname_list) == len(score_list)):
        min_len = min(len(trial_lines), len(fname_list), len(score_list))
        print(f"Mismatch detected! Aligned to minimum length: {min_len}")
        trial_lines = trial_lines[:min_len]
        fname_list = fname_list[:min_len]
        score_list = score_list[:min_len]
    with open(save_path, "w", encoding="utf-8") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            parts = trl.strip().split(' ')
            if len(parts) < 5:
                continue
            _, utt_id, _, src, key = parts
            assert fn == utt_id, f"Mismatch utterance: loader returned {fn} but trial has {utt_id}"
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))


def augment_audio(batch: torch.Tensor, noise_factor_range) -> torch.Tensor:
    noise_factor = random.uniform(*noise_factor_range)
    noise = torch.randn_like(batch) * noise_factor
    return batch + noise


def train_epoch(trn_loader: DataLoader, model, optim, device: torch.device,
                config: dict, grad_clip: float = 5.0) -> float:
    running_loss = 0.0
    num_total = 0.0
    model.train()
    loss_weight = config.get("cfad_loss_weight", [0.5, 0.5]) if config.get("use_cfad", False) else config.get("la_loss_weight", [0.1, 0.9])
    weight = torch.FloatTensor(loss_weight).to(device)
    criterion = FocalLoss(gamma=2.0, weight=weight) if config.get("use_focal", False) else nn.CrossEntropyLoss(weight=weight)
    scaler = torch.cuda.amp.GradScaler()
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        try:
            batch_y = [convert_label(label) for label in batch_y]
        except Exception as e:
            raise ValueError("Label conversion error: " + str(e))
        batch_y = torch.tensor(batch_y).view(-1).long().to(device)
        batch_x = extra_data_augment(batch_x, config)
        if config.get("use_aug", False):
            noise_range = config.get("aug_noise_factor", [0.001, 0.005])
            batch_x = augment_audio(batch_x, noise_factor_range=noise_range)
        optim.zero_grad()
        with torch.cuda.amp.autocast():
            _, logits = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: logits contain NaNs/Infs before loss; clamping them.")
            logits = torch.clamp(logits, -1e6, 1e6)
            batch_loss = criterion(logits, batch_y)
        scaler.scale(batch_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optim)
        scaler.update()
        running_loss += batch_loss.item() * batch_size
    running_loss /= num_total
    return running_loss


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f_json:
        config = json.load(f_json)
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    config.setdefault("eval_all_best", "True")
    config.setdefault("freq_aug", "False")
    set_seed(args.seed, config)
    la_database_path = Path(config["database_path"])
    if config.get("use_cfad", False) and any(stage in config.get("run_cfad", []) for stage in ["eval", "dev", "train"]):
        dev_trial_path = Path(config["cfad_dev_protocol"])
        eval_trial_path = Path(config["cfad_eval_protocol"])
        train_trial_path = Path(config["cfad_train_protocol"])
    else:
        prefix = f"ASVspoof2019.{track}"
        dev_trial_path = la_database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix}.cm.dev.trl.txt"
        eval_trial_path = la_database_path / f"ASVspoof2019_{track}_cm_protocols/{prefix}.cm.eval.trl.txt"
    output_dir = Path(args.output_dir)
    model_tag = f"{track}_{os.path.splitext(os.path.basename(args.config))[0]}_ep{config['num_epochs']}_bs{config['batch_size']}"
    if args.comment:
        model_tag = f"{model_tag}_{args.comment}"
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(str(model_tag))
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")
    model = get_model(model_config, device)
    trn_loader, dev_loader, eval_loader = get_loader(la_database_path, args.seed, config)
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    if config.get("use_warmup", False):
        warmup_epochs = config.get("warmup_epochs", 5)
        scheduler = get_warmup_scheduler(optimizer, warmup_epochs, config["num_epochs"])
    optimizer_swa = SWA(optimizer)
    if args.eval:
        model.load_state_dict(torch.load(config["model_path"], map_location=device), strict=False)
        print("Model loaded: {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device, str(eval_score_path), str(eval_trial_path))
        calculate_tDCF_EER(cm_scores_file=str(eval_score_path),
                           asv_score_file=la_database_path / config["asv_score_path"],
                           output_file=str(model_tag / "t-DCF_EER.txt"))
        print("DONE.")
        sys.exit(0)
    best_dev_eer = 1.0
    best_eval_eer = 100.0
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.0
    n_swa_update = 0
    with open(model_tag / "metric_log.txt", "a") as f_log:
        f_log.write("=" * 5 + "\n")
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device, config)
        scheduler.step()
        produce_evaluation_file(dev_loader, model, device,
                                str(metric_path / "dev_score.txt"), str(dev_trial_path))
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=str(metric_path / "dev_score.txt"),
            asv_score_file=la_database_path / config["asv_score_path"],
            output_file=str(metric_path / f"dev_t-DCF_EER_{epoch:03d}epo.txt"),
            printout=False
        )
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf: {:.5f}".format(running_loss, dev_eer, dev_tdcf))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)
        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print("Best model found at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(), model_save_path / f"epoch_{epoch}_{dev_eer:03.3f}.pth")
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device, str(eval_score_path), str(eval_trial_path))
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=str(eval_score_path),
                    asv_score_file=la_database_path / config["asv_score_path"],
                    output_file=str(metric_path / f"t-DCF_EER_{epoch:03d}epo.txt")
                )
                log_text = f"epoch{epoch:03d}, "
                if eval_eer < best_eval_eer:
                    log_text += f"best eer, {eval_eer:.4f}%"
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += f" best tdcf, {eval_tdcf:.4f}"
                    best_eval_tdcf = eval_tdcf
                    torch.save(model.state_dict(), model_save_path / "best.pth")
                if log_text:
                    print(log_text)
                    with open(model_tag / "metric_log.txt", "a") as f_log:
                        f_log.write(log_text + "\n")
            print("Saving epoch {} for SWA".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)
    print("Start final evaluation")
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, str(eval_score_path), str(eval_trial_path))
    eval_eer, eval_tdcf = calculate_tDCF_EER(
        cm_scores_file=str(eval_score_path),
        asv_score_file=la_database_path / config["asv_score_path"],
        output_file=str(model_tag / "t-DCF_EER.txt")
    )
    with open(model_tag / "metric_log.txt", "a") as f_log:
        f_log.write("=" * 5 + "\n")
        f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    torch.save(model.state_dict(), model_save_path / "swa.pth")
    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(), model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(best_eval_eer, best_eval_tdcf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config", dest="config", type=str, required=True, help="Configuration file")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./exp_result", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--eval", action="store_true", help="Evaluate the given model and exit")
    parser.add_argument("--comment", type=str, default=None, help="Comment to describe the saved model")
    parser.add_argument("--eval_model_weights", type=str, default=None, help="Directory to the model weight file")
    main(parser.parse_args())