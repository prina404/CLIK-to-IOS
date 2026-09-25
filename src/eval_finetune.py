import argparse
import json
import multiprocessing
import os

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree

from s1_oral_scan import CMP_DIR, DATASET_DIR, DIFFUSION_CKPT, RUNS_DIR, load_teeth
import model.core_util as util
from infer_oral_scan import infer, is_done

METRICS = ["rotation [deg]", "translation [mm]", "point cloud [mm]", "collision [mm]"]


def signed_distance(mesh, points):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices), o3d.core.float32),
                        o3d.core.Tensor(np.asarray(mesh.faces), o3d.core.uint32))
    return scene.compute_signed_distance(o3d.core.Tensor(points, o3d.core.float32)).numpy()


def collision_error(teeth):
    depth, count = 0.0, 0
    for arch in (range(2, 16), range(18, 32)):
        present = [t for t in arch if t in teeth]
        for a, b in zip(present, present[1:]):
            for src, dst in ((a, b), (b, a)):
                # only vertices inside both bounding boxes can be inside the neighbour
                lo = np.maximum(teeth[src].bounds[0], teeth[dst].bounds[0])
                hi = np.minimum(teeth[src].bounds[1], teeth[dst].bounds[1])
                v = np.asarray(teeth[src].vertices)
                v = v[((v >= lo) & (v <= hi)).all(1)]
                if len(v):
                    sdf = signed_distance(teeth[dst], v)  # negative inside
                    depth -= sdf[sdf < 0].sum()
                    count += (sdf < 0).sum()
    return depth / max(count, 1)


def evaluate_patient(patient, result_dir):
    ori = load_teeth(DATASET_DIR / patient, "Ori")
    final = load_teeth(DATASET_DIR / patient, "Final")
    pred = {int(f.stem): trimesh.load(f) for f in (result_dir / patient / "results").glob("*.ply")}

    rot, trans, pc = [], [], []
    for tooth_id in ori:
        ori_v, gt_v, pred_v = (torch.from_numpy(np.asarray(m[tooth_id].vertices)) for m in (ori, final, pred))
        if len(gt_v) != len(ori_v):  # a few teeth are re-segmented in final: match vertices by proximity
            gt_v = gt_v[cKDTree(gt_v).query(ori_v)[1]]
        R_gt, T_gt = util.solve_rigid_matrix(ori_v, gt_v)
        R_pred, T_pred = util.solve_rigid_matrix(ori_v, pred_v)
        rot.append(util.rot_mat_error(R_gt, R_pred).item())
        trans.append((T_gt - T_pred).abs().sum().item() / 3)
        pc.append((gt_v - pred_v).norm(dim=1).mean().item())
    return np.mean(rot), np.mean(trans), np.mean(pc), collision_error(pred)


def evaluate(patients, result_dir):
    os.environ["OMP_NUM_THREADS"] = "1"  # one thread per worker, or the 8 workers oversubscribe the CPU
    with multiprocessing.get_context("spawn").Pool() as pool:
        rows = pool.starmap(evaluate_patient, [(p, result_dir) for p in patients])
    return pd.DataFrame(rows, index=patients, columns=METRICS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', help="finetuning run to evaluate, e.g. version_15 (default: the latest)")
    args = parser.parse_args()

    runs = RUNS_DIR / "lightning_logs"
    run_dir = runs / args.run if args.run else max(runs.glob("*/finetuned.pth"), key=lambda f: f.stat().st_mtime).parent
    patients = json.loads((run_dir / "splits.json").read_text())["test"]

    metrics = {}
    for name, ckpt, out_dir in (("released", DIFFUSION_CKPT, CMP_DIR / "baseline"),
                                ("finetuned", run_dir / "finetuned.pth", CMP_DIR / run_dir.name)):
        print(f"\n{name}: {ckpt}")
        infer([p for p in patients if not is_done(p, out_dir)], out_dir, ckpt)
        metrics[name] = evaluate(patients, out_dir)
    pd.concat(metrics, axis=1).to_csv(CMP_DIR / f"{run_dir.name}_metrics.csv")

    print(f"\n{run_dir.name}, {len(patients)} held-out patients")
    print(pd.DataFrame({name: m.mean().map("{:.3f}".format) + " ± " + m.std().map("{:.3f}".format)
                        for name, m in metrics.items()}).T.to_string())
