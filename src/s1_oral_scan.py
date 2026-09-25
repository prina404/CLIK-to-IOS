import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "src" / "CLIK-Diffusion" / "Code"
sys.path.append(str(CODE_DIR))  # upstream modules: model.*, s2_LandmarkDiffusion, s3_SolveMatrix
from model.point_mlp import PointMLP

DATASET_DIR = ROOT / "data" / "Orthodontic_dental_dataset"
DIFFUSION_CKPT = CODE_DIR / "checkpoint" / "diffusion-e20000.pth"
OUTPUT_DIR = ROOT / "outputs" / "Output"
CMP_DIR = ROOT / "outputs" / "Output_cmp"
CACHE_DIR = ROOT / "outputs" / "finetune_cache"
RUNS_DIR = ROOT / "outputs" / "finetune_runs"

# detection head -> (checkpoint, UTN ids of its teeth, landmark ids it predicts)
HEADS = {
    "incisor": ("[Crown]incisor-e837.pt", [7, 8, 9, 10, 23, 24, 25, 26], ['3', '4', '13', '15', '16', '34', '35']),
    "cuspid": ("[Crown]cuspid-e879.pt", [6, 11, 22, 27], ['3', '4', '5', '19', '20', '34', '35']),
    "premolar": ("[Crown]premolar-e443.pt", [4, 5, 12, 13, 20, 21, 28, 29], ['4', '6', '7', '21', '22', '23', '34', '35']),
    "molar": ("[Crown]molar-e178.pt", [2, 3, 14, 15, 18, 19, 30, 31], ['8', '9', '10', '11', '21', '22', '25', '30', '34', '35']),
}

# FDI -> UTN numbering. Third molars are not used by CLIK.
FDI_TO_UTN = {
    "17": 2, "16": 3, "15": 4, "14": 5, "13": 6, "12": 7, "11": 8,
    "21": 9, "22": 10, "23": 11, "24": 12, "25": 13, "26": 14, "27": 15,
    "37": 18, "36": 19, "35": 20, "34": 21, "33": 22, "32": 23, "31": 24,
    "41": 25, "42": 26, "43": 27, "44": 28, "45": 29, "46": 30, "47": 31,
}

# The checkpoints expect the CBCT jaw frame: patient-right +y, anterior -x, upper arch +z, dentition
# centred at CANON_CENTROID. Patients 0001-0277 are stored patient-right +x, anterior -y, upper arch -z;
# from 0278 on they are also rotated 180 degrees about the left-right axis.
R_ALIGN = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
R_FLIP = np.diag([-1., 1., -1.])
CANON_CENTROID = np.array([-10., 0., 0.])


def read_segmentation(json_path):
    with open(json_path) as f:
        segmentation = json.load(f)["segmentation"]
    return {FDI_TO_UTN[fdi]: np.array(tooth["vertices"]) for fdi, tooth in segmentation.items() if fdi in FDI_TO_UTN}


def is_extraction(patient_dir):
    ori = {fdi for arch in "UL" for fdi in json.load(open(patient_dir / "ori" / f"{arch}_Ori.json"))["segmentation"]}
    final = {fdi for arch in "UL" for fdi in json.load(open(patient_dir / "final" / f"{arch}_Final.json"))["segmentation"]}
    return ori != final


def patient_frame(patient_dir):
    R = R_FLIP @ R_ALIGN if int(patient_dir.name) >= 278 else R_ALIGN
    vertices = np.concatenate([v for arch in "UL" for v in read_segmentation(patient_dir / "ori" / f"{arch}_Ori.json").values()])
    return R, CANON_CENTROID - (vertices @ R.T).mean(0)


def load_teeth(patient_dir, stem="Ori"):
    R, t = patient_frame(patient_dir)  # always from ori: final must not be recentred on its moved teeth
    teeth = {}
    for arch in "UL":
        scan = patient_dir / stem.lower() / f"{arch}_{stem}"
        mesh = trimesh.load(scan.with_suffix(".stl"), process=True, validate=True)
        tree = cKDTree(mesh.vertices)
        mesh.vertices = mesh.vertices @ R.T + t
        for tooth_id, vertices in read_segmentation(scan.with_suffix(".json")).items():
            selected = np.zeros(len(mesh.vertices), dtype=bool)
            selected[tree.query(vertices, workers=-1)[1]] = True
            teeth[tooth_id] = mesh.submesh([np.flatnonzero(selected[mesh.faces].all(1))], append=True)
    return teeth


def load_detection():
    models = {}
    for name, (ckpt, _, landmark_ids) in HEADS.items():
        model = PointMLP(num_classes=len(landmark_ids), points=2048, embed_dim=128)
        model.load_state_dict(torch.load(CODE_DIR / "checkpoint" / ckpt)['model'])
        models[name] = model.cuda().eval()
    return models


def pc_normalize(pc):
    c = np.mean(pc, axis=0)
    pc = pc - c
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    return pc / m, c, m


def farthest_point_sample(clouds, npoint):
    # every tooth at once: shorter clouds are padded by repeating their own points, which are never picked twice
    lengths = np.array([len(c) for c in clouds])
    xyz = torch.from_numpy(np.stack([c[np.arange(lengths.max()) % len(c)] for c in clouds])).cuda()
    batch = torch.arange(len(clouds), device="cuda")
    centroids = torch.zeros(len(clouds), npoint, dtype=torch.long, device="cuda")
    distance = torch.full(xyz.shape[:2], 1e10, dtype=xyz.dtype, device="cuda")
    farthest = torch.randint(0, xyz.shape[1], (len(clouds),), device="cuda")
    for i in range(npoint):
        centroids[:, i] = farthest
        distance = torch.minimum(distance, ((xyz - xyz[batch, farthest].unsqueeze(1)) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return centroids.cpu().numpy() % lengths[:, None]


def detect_landmarks(teeth, models, num_samples=2048):
    ids = sorted(teeth)
    points, centers, scales = zip(*[pc_normalize(teeth[t].vertices) for t in ids])
    samples = farthest_point_sample(points, num_samples)
    landmarks = {}
    for name, (_, head_teeth, landmark_ids) in HEADS.items():
        group = [i for i, t in enumerate(ids) if t in head_teeth]
        if not group:  # a few patients have no molars or no cuspids
            continue
        sampled = np.stack([points[i][samples[i]] for i in group])                         # (G, 2048, 3)
        normals = np.stack([teeth[ids[i]].vertex_normals[samples[i]] for i in group])      # (G, 2048, 3)
        with torch.no_grad():
            pred = models[name](torch.from_numpy(sampled).float().cuda().transpose(1, 2),
                                torch.from_numpy(normals).float().cuda().transpose(1, 2),
                                torch.tensor([ids[i] for i in group]).cuda())             # (G, 2048, num_landmarks)
        for k, i in enumerate(group):
            coords = sampled[k][pred[k].argmax(0).cpu().numpy()] * scales[i] + centers[i]  # back to mm
            landmarks[ids[i]] = {'0': teeth[ids[i]].vertices.mean(0), **dict(zip(landmark_ids, coords))}
    return landmarks
