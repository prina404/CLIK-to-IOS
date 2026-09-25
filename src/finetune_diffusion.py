import argparse
import json
import multiprocessing
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from s1_oral_scan import (CACHE_DIR, DATASET_DIR, DIFFUSION_CKPT, RUNS_DIR, detect_landmarks, is_extraction,
                          load_detection, load_teeth, patient_frame, read_segmentation)
import model.core_util as util
from model.diffusion_network import Network_1dUNet
from s2_LandmarkDiffusion import diffusion_cfgs, organize_input, query_points


# ================================ Data: stage-1 landmarks (X) and where each tooth moves them (Y) ================================
EXTRACTIONS = CACHE_DIR / "extractions.json"  # patient -> whether it is an extraction case, which is slow to check


def init_worker():  # every caching process runs its own copy of the landmark detection networks
    global MODELS
    MODELS = load_detection()


def cache_patient(patient_dir):
    L.seed_everything(int(patient_dir.name), verbose=False)  # point sampling is random: same cache whichever process or order
    teeth = load_teeth(patient_dir)
    landmarks = detect_landmarks(teeth, MODELS)
    R_frame, t_frame = patient_frame(patient_dir)
    moved = {}
    for arch in "UL":
        ori = read_segmentation(patient_dir / "ori" / f"{arch}_Ori.json")
        final = read_segmentation(patient_dir / "final" / f"{arch}_Final.json")
        for tooth_id in ori:
            ori_v, final_v = ori[tooth_id] @ R_frame.T + t_frame, final[tooth_id] @ R_frame.T + t_frame
            if len(ori_v) != len(final_v):  # re-segmented tooth: no vertex correspondence, register with ICP
                final_v = trimesh.registration.icp(ori_v, final_v, max_iterations=50)[1]
            R, T = util.solve_rigid_matrix(torch.from_numpy(ori_v), torch.from_numpy(final_v))
            moved[tooth_id] = {k: R.numpy() @ v + T.numpy() for k, v in landmarks[tooth_id].items()}

    cond = organize_input(landmarks, teeth)
    np.savez(CACHE_DIR / f"{patient_dir.name}.npz",
             cond=cond, descriptor=query_points(cond, teeth), target=organize_input(moved, teeth)[:, :3])


def build_cache(workers):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    patients = sorted(DATASET_DIR.iterdir())
    uncached = [p for p in patients if not (CACHE_DIR / f"{p.name}.npz").exists()]
    extraction = json.loads(EXTRACTIONS.read_text()) if EXTRACTIONS.exists() else {}
    unscreened = [p for p in uncached if p.name not in extraction]
    if unscreened:
        extraction |= {p.name: is_extraction(p) for p in tqdm(unscreened, desc="Screening for extractions")}
        EXTRACTIONS.write_text(json.dumps(extraction, indent=2, sort_keys=True))
    todo = [p for p in uncached if not extraction[p.name]]
    print(f"{len(patients)} patients: {len(patients) - len(uncached)} already cached, "
          f"{len(uncached) - len(todo)} extraction cases skipped, {len(todo)} to cache")
    if todo:
        print(f"Starting {workers} caching processes")
        with multiprocessing.get_context("spawn").Pool(workers, init_worker) as pool:  # spawn: each process sets up CUDA
            list(tqdm(pool.imap_unordered(cache_patient, todo), total=len(todo), desc="Caching patients"))


def load_split(name, patients):
    samples = [dict(np.load(CACHE_DIR / f"{p}.npz")) for p in tqdm(patients, desc=f"Loading {name} split")]
    return TensorDataset(*[torch.from_numpy(np.stack([s[key] for s in samples])).float().transpose(1, 2)
                           for key in ("cond", "descriptor", "target")])  # (N, 5, 256), (N, 384, 256), (N, 3, 256)


# ================================ Loss: diffusion MSE + the paper's arch (Eq. 4) and individual-tooth (Eq. 6) terms ================================
LANDMARK_ID = np.rint(organize_input({}, {})[:, 3] * 35)  # landmark id held by each of the 256 slots
TOOTH_INDEX = np.concatenate([np.arange(28), np.repeat(np.arange(28), np.diff(util.landmark_slices))])  # and its tooth


def tooth_average(selected):  # (28, 256) matrix: matrix @ points averages each tooth's selected slots
    mask = (TOOTH_INDEX == np.arange(28)[:, None]) & selected
    return torch.tensor(mask / mask.sum(1, keepdims=True), dtype=torch.float32)


TOOTH_CENTRE = tooth_average(np.ones(256, dtype=bool))
CROWN_TIP = tooth_average(np.isin(LANDMARK_ID, [5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 19, 20]))  # cusps and incisal edges
MESIAL_DISTAL = tooth_average(LANDMARK_ID == 34) - tooth_average(LANDMARK_ID == 35)


def arch_coefficients(points, present):  # 4th-order polynomial x = f(y) through the tooth centres of each arch
    centres = (TOOTH_CENTRE.to(points) @ points).double()
    return torch.stack([util.polyfit_weighted(centres[:, arch, 1], centres[:, arch, 0], present[:, arch].double())
                        for arch in (slice(0, 14), slice(14, 28))], 1)


def tooth_axes(points):  # root-crown and mesial-distal direction of every tooth
    return CROWN_TIP.to(points) @ points - points[:, :28], MESIAL_DISTAL.to(points) @ points


def full_loss(pred, target, cond):
    present = (cond[:, :3, :28] != 0).any(1).float()  # (B, 28), absent teeth are zero-filled
    pred, target = pred.transpose(1, 2), target.transpose(1, 2)  # (B, 256, 3)

    arch = ((arch_coefficients(pred, present) - arch_coefficients(target, present)) ** 2).sum(-1).mean(0).sum()
    individual = 0
    for axis_pred, axis_target in zip(tooth_axes(pred), tooth_axes(target)):
        cosine = F.cosine_similarity(axis_pred, axis_target, dim=-1)
        defined = present * (axis_target.norm(dim=-1) > 1e-4)  # stage 1 sometimes puts both contact points on one vertex
        individual = individual + (((1 - cosine) * defined).sum(1) / defined.sum(1)).mean()
    return F.mse_loss(pred, target) + 5e-5 * arch + 0.01 * individual


# ================================ Model ================================
class Finetuner(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = Network_1dUNet(diffusion_cfgs['unet'], diffusion_cfgs['beta_schedule'])
        self.net.load_state_dict(torch.load(DIFFUSION_CKPT, map_location="cpu"), strict=False)
        self.net.set_new_noise_schedule(device=torch.device("cpu"))

    def denoising_loss(self, batch, t, generator=None):
        cond, descriptor, y_0 = batch
        gamma_prev, gamma_t = self.net.gammas[t - 1], self.net.gammas[t]
        gamma = gamma_prev + (gamma_t - gamma_prev) * torch.rand(len(t), generator=generator, device=self.device)
        gamma = gamma.view(-1, 1, 1)
        y_t = gamma.sqrt() * y_0 + (1 - gamma).sqrt() * torch.randn(y_0.shape, generator=generator, device=self.device)
        y_0_hat = self.net.denoise_fn(torch.cat([cond, y_t], dim=1), gamma.view(-1, 1), descriptor)  # predicts y_0, not the noise
        return full_loss(y_0_hat, y_0, cond)

    def training_step(self, batch, batch_idx):
        t = torch.randint(1, self.net.num_timesteps, (len(batch[0]),), device=self.device)
        loss = self.denoising_loss(batch, t)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # fixed timesteps and noise, so that val_loss is comparable between epochs
        t = torch.linspace(1, self.net.num_timesteps - 1, len(batch[0]), device=self.device).long()
        generator = torch.Generator(self.device).manual_seed(batch_idx)
        self.log("val_loss", self.denoising_loss(batch, t, generator), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=15, help="stop after this many epochs without a better val_loss")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--wandb', action="store_true", help="also log to Weights & Biases")
    parser.add_argument('--workers', type=int, default=2,
                        help="processes caching patients in parallel, each needs ~2.6 GB of GPU memory for landmark detection")
    args = parser.parse_args()

    L.seed_everything(0)
    build_cache(args.workers)
    patients = sorted(p.stem for p in CACHE_DIR.glob("*.npz"))
    random.Random(0).shuffle(patients)
    n = round(len(patients) / 10)
    splits = {"val": sorted(patients[:n]), "test": sorted(patients[n:2 * n]), "train": sorted(patients[2 * n:])}

    loggers = [CSVLogger(RUNS_DIR)] + ([WandbLogger(project="clikt_diffusion_finetune", save_dir=RUNS_DIR)] if args.wandb else [])
    checkpoint = ModelCheckpoint(monitor="val_loss")
    trainer = L.Trainer(max_epochs=args.max_epochs, logger=loggers, log_every_n_steps=10,
                        callbacks=[checkpoint, EarlyStopping("val_loss", patience=args.patience)])
    run_dir = Path(trainer.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "splits.json").write_text(json.dumps(splits, indent=2))  # eval_finetune.py scores the test split

    train = DataLoader(load_split("train", splits["train"]), args.batch_size, shuffle=True, drop_last=True)
    trainer.fit(Finetuner(args.lr), train, DataLoader(load_split("val", splits["val"]), args.batch_size))

    torch.save(Finetuner.load_from_checkpoint(checkpoint.best_model_path).net.state_dict(), run_dir / "finetuned.pth")
    print(f"Finetuned weights: {run_dir / 'finetuned.pth'}")
