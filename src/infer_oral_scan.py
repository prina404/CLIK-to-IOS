import argparse

import numpy as np
import torch

from s1_oral_scan import DATASET_DIR, DIFFUSION_CKPT, OUTPUT_DIR, detect_landmarks, is_extraction, load_detection, load_teeth
from model.core_util import set_seed
from s2_LandmarkDiffusion import load_diffusion, organize_input, query_points
from s3_SolveMatrix import save_mesh, save_transformation, solve_and_trans_mesh


def is_done(patient, save_dir):
    return (save_dir / patient / "results" / "transformation.json").exists()


def infer(patients, save_dir, diffusion_ckpt=DIFFUSION_CKPT, chunk_size=16):
    todo = [p for p in patients if not is_extraction(DATASET_DIR / p)]
    print(f"{len(todo)} patients to process, {len(patients) - len(todo)} extraction cases skipped")
    if not todo:
        return
    set_seed(1)
    models, network = load_detection(), load_diffusion(diffusion_ckpt)

    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        print(f"Patients {start + 1}-{start + len(chunk)} of {len(todo)}")
        # ================================ Stage1: landmark detection ================================
        teeth = [load_teeth(DATASET_DIR / p) for p in chunk]
        inputs = [organize_input(detect_landmarks(t, models), t) for t in teeth]  # np(256, 5)
        descriptors = [query_points(x, t) for x, t in zip(inputs, teeth)]         # np(256, 384)

        # ================================ Stage2: diffusion, whole chunk at once ================================
        y_cond = torch.from_numpy(np.stack(inputs)).float().transpose(1, 2).cuda()           # (B, 5, 256)
        features = torch.from_numpy(np.stack(descriptors)).float().transpose(1, 2).cuda()    # (B, 384, 256)
        with torch.no_grad():
            y_pred, _ = network.restoration(y_cond, y_t=torch.rand_like(y_cond[:, :3]), sample_num=1, extra_features=features)

        # ================================ Stage3: rigid transformation of each tooth ================================
        for i, patient in enumerate(chunk):
            meshes, transformations = solve_and_trans_mesh(y_pred[i:i + 1], y_cond[i:i + 1], teeth[i])
            save_mesh(meshes, patient, save_dir)
            save_transformation(transformations, patient, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--patient', default="0007", help="patient to predict")
    parser.add_argument('-b', '--batch', action="store_true", help="predict every patient, resuming a previous run")
    parser.add_argument('--ckpt', default=DIFFUSION_CKPT, help="diffusion weights, e.g. a finetuned.pth (default: released)")
    args = parser.parse_args()

    patients = [p.name for p in sorted(DATASET_DIR.iterdir()) if not is_done(p.name, OUTPUT_DIR)] if args.batch else [args.patient]
    infer(patients, OUTPUT_DIR, args.ckpt)
