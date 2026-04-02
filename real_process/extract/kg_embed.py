#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_kg_training.py

- 每个 run 单独训练并保存 KG 嵌入（run_{id}/kg/）
- 删除 joint-embedding 逻辑（不合并跨 run）
- 提供安全的 load_run_kg_embeddings() 供下游读取
- 输出文件： run_{id}/kg/kg_embeddings.npz, run_{id}/kg/{MODEL}_metrics.json, run_{id}/kg/{MODEL}_model/...
"""

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import numpy as np
import torch
import random
import pandas as pd
from pathlib import Path
import logging
import os
import json
import time
from tqdm import tqdm
from typing import Tuple, Dict, Optional

dir_path = "scratch/project/path_planning/kg_results"
os.makedirs(dir_path, exist_ok=True)

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scratch/project/path_planning/kg_results/kg_embedding_batch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
DEFAULT_TRAINING = {
    "loss": "softplus",
    "loss_kwargs": None,
    "regularizer": "lp",
    "regularizer_kwargs": {"p": 2, "weight": 1e-6},
}

# ---------- model configs ----------
MODEL_CONFIGS = {
    "DistMult": {"model": "DistMult", "default_dim": 100},
    "ComplEx":  {"model": "ComplEx",  "default_dim": 150},
    "TransE":   {"model": "TransE",   "default_dim": 16},
    "RotatE":   {"model": "RotatE",   "default_dim": 200},
    "PairRE":   {"model": "PairRE",   "default_dim": 200},
}

# ---------- helper: safe loader for saved npz embeddings ----------
def safe_load_kg_npz(npz_path: str) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Load kg_embeddings.npz produced by this pipeline (or similar).
    Return (embeddings_matrix (entities), mapping_dict)
    mapping_dict: entity_label -> id (as stored)
    This function is robust to different internal key names used historically:
      - entity_embeddings / entity_emb / ent_emb
      - entity_mapping / entity_ids / entity_id_map (array-of-pairs or dict)
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        logger.warning(f"Unable to load npz {npz_path}: {e}")
        return None, {}

    # try entity embeddings
    ent_keys_candidates = ['entity_embeddings', 'entities', 'entity_emb', 'ent_embeddings']
    rel_keys_candidates = ['relation_embeddings', 'relations', 'relation_emb']

    ent_emb = None
    rel_emb = None
    for k in ent_keys_candidates:
        if k in data:
            ent_emb = data[k]
            break
    for k in rel_keys_candidates:
        if k in data:
            rel_emb = data[k]
            break

    # try mapping
    mapping = {}
    # common names we might have used
    mapping_keys = ['entity_mapping', 'entity_ids', 'entity_id_map', 'entities_mapping']
    for k in mapping_keys:
        if k in data:
            raw = data[k]
            # raw might be array of pairs, or dict-like saved as object
            try:
                # if array of pairs [[id,label],...]
                if isinstance(raw, np.ndarray) and raw.dtype == object:
                    for pair in np.atleast_1d(raw):
                        if len(pair) == 2:
                            # pair could be (id,label) or (label,id)
                            a, b = pair
                            # detect which is numeric id
                            if isinstance(a, (int, np.integer, str)) and isinstance(b, (str, bytes)):
                                if isinstance(a, bytes):
                                    a = a.decode()
                                mapping[str(b)] = int(a) if str(a).isdigit() else a
                            elif isinstance(b, (int, np.integer, str)) and isinstance(a, (str, bytes)):
                                if isinstance(b, bytes):
                                    b = b.decode()
                                mapping[str(a)] = int(b) if str(b).isdigit() else b
                            else:
                                # fallback: string-string
                                mapping[str(pair[0])] = pair[1]
                elif isinstance(raw.tolist(), dict):
                    mapping = dict(raw.tolist())
                else:
                    # if it's a dict-like object
                    mapping = dict(raw)
            except Exception:
                # last resort: try iterate
                try:
                    for item in list(raw):
                        if len(item) == 2:
                            mapping[str(item[0])] = item[1]
                except Exception:
                    mapping = {}
            break

    # Also accept direct dict saved as npz entry:
    if not mapping:
        # check for any object arrays that look like mapping
        for k in data.files:
            val = data[k]
            if isinstance(val, np.ndarray) and val.dtype == object and val.shape[-1] == 2:
                try:
                    cand = {}
                    for p in val:
                        cand[str(p[0])] = p[1]
                    # heuristic: many keys but at least one key looks like an entity label
                    if len(cand) > 0:
                        mapping = cand
                        break
                except Exception:
                    continue

    # final normalization: ensure mapping keys are strings and values are ints (if numeric)
    normalized = {}
    for k, v in mapping.items():
        ks = str(k)
        try:
            vi = int(v)
        except Exception:
            try:
                vi = int(str(v))
            except Exception:
                vi = v
        normalized[ks] = vi
    mapping = normalized

    return ent_emb, mapping

# ---------- load_run_kg_embeddings (safe, used by GAT input pipeline) ----------
def load_run_kg_embeddings(run_id: int, base_dir: str) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Look for run_{run_id}/kg/kg_embeddings.npz and return (entity_emb_matrix, entity_to_id mapping).
    Return (None, {}) if not present or unreadable.

    entity_to_id mapping returned as dict { entity_label_str: int_index_in_matrix }
    """
    run_dir = Path(base_dir) / f"run_{run_id}" / "kg"
    npz_path = run_dir / "kg_embeddings.npz"
    if not npz_path.exists():
        logger.info(f"No KG embeddings file for run{run_id} at {npz_path}")
        return None, {}
    ent_emb, mapping = safe_load_kg_npz(str(npz_path))
    # If mapping values are labels->indices, done. If mapping is idx->label invert it.
    if mapping:
        # detect if mapping currently is like {label: idx} or {idx: label}
        sample_vals = list(mapping.values())
        sample_keys = list(mapping.keys())
        if len(sample_vals) > 0:
            # if values are strings and keys look numeric => invert
            if all(isinstance(v, (str, bytes)) for v in sample_vals) and all(str(k).isdigit() for k in sample_keys):
                inv = {}
                for k, v in mapping.items():
                    inv[str(v)] = int(k)
                mapping = inv
        # ensure keys are str and values int
        final_map = {}
        for k, v in mapping.items():
            try:
                final_map[str(k)] = int(v)
            except Exception:
                try:
                    final_map[str(k)] = int(str(v))
                except Exception:
                    # can't coerce, skip
                    continue
        mapping = final_map
    return ent_emb, mapping

# ---------- single-run training ----------
def train_single_run_kg(
    data_path: str,
    output_dir: str,
    model_name: str = "ComplEx",
    embedding_dim: int = None,
    num_epochs: int = 80,
    random_seed: int = 42,
    batch_size: int = 64
) -> Path:
    """
    Train KG embeddings for a single run and save results to output_dir (which should be run_{id}/kg).
    Returns path to saved kg_embeddings.npz
    """
    run_id = Path(data_path).parent.name.split("_")[-1]  # data_path=.../run_X/triples.tsv
    model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["ComplEx"])
    if embedding_dim is None:
        embedding_dim = model_config.get("default_dim", 50)

    logger.info(f"Run {run_id}: KG training start | model={model_name} dim={embedding_dim}")

    # reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Triples file not found: {data_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load triples
    try:
        df = pd.read_csv(data_path, sep='\t', header=None, names=['head', 'relation', 'tail'], dtype=str)
        triples = df[['head', 'relation', 'tail']].values
        tf = TriplesFactory.from_labeled_triples(triples=triples, create_inverse_triples=False)
        logger.info(f"Run {run_id}: triples loaded: {len(triples)} | entities: {tf.num_entities}, relations: {tf.num_relations}")
    except Exception as e:
        logger.exception(f"Run {run_id}: failed to load triples: {e}")
        raise

    # split
    training, validation, testing = tf.split(ratios=[0.8, 0.1, 0.1], random_state=random_seed)

    model_kwargs = {
        "embedding_dim": embedding_dim,
        "random_seed": random_seed,
    }
    training_kwargs = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "checkpoint_frequency": max(1, min(10, num_epochs // 10))
    }

    pipeline_model_arg = model_config["model"]

    try:
        start_time = time.time()
        loss = model_config.get("loss", DEFAULT_TRAINING["loss"])
        loss_kwargs = model_config.get("loss_kwargs", DEFAULT_TRAINING["loss_kwargs"])
        regularizer = model_config.get("regularizer", DEFAULT_TRAINING["regularizer"])
        regularizer_kwargs = model_config.get("regularizer_kwargs", DEFAULT_TRAINING["regularizer_kwargs"])

        result = pipeline(
            training=training,
            validation=validation,
            testing=testing,
            model=pipeline_model_arg,
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            loss=loss,
            loss_kwargs=loss_kwargs,
            regularizer=regularizer,
            regularizer_kwargs=regularizer_kwargs,
            evaluator_kwargs={"filtered": True},
            random_seed=random_seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_tqdm=False
        )
        train_time = time.time() - start_time
        logger.info(f"Run {run_id}: training finished in {train_time:.2f}s")
    except Exception as e:
        logger.exception(f"Run {run_id}: training failed: {e}")
        raise

    # save model and embeddings into output_dir
    try:
        model_dir = output_dir / f"{pipeline_model_arg}_model"
        result.save_to_directory(model_dir)

        # extract embeddings
        ent_emb = result.model.entity_representations[0]().detach().cpu().numpy()
        rel_emb = result.model.relation_representations[0]().detach().cpu().numpy()

        # tf.entity_to_id is mapping label->id
        entity_to_id = {str(k): int(v) for k, v in tf.entity_to_id.items()}
        relation_to_id = {str(k): int(v) for k, v in tf.relation_to_id.items()}

        kg_npz = output_dir / "kg_embeddings.npz"
        np.savez(
            str(kg_npz),
            entity_embeddings=ent_emb,
            relation_embeddings=rel_emb,
            entity_mapping=np.array(list(entity_to_id.items()), dtype=object),
            relation_mapping=np.array(list(relation_to_id.items()), dtype=object),
            metadata={
                "run_id": str(run_id),
                "model": pipeline_model_arg,
                "embedding_dim": embedding_dim,
                "num_epochs": num_epochs,
                "random_seed": random_seed,
                "train_time_s": train_time,
                "num_entities": int(tf.num_entities),
                "num_relations": int(tf.num_relations),
                "num_triples": int(tf.num_triples)
            }
        )
        logger.info(f"Run {run_id}: kg_embeddings saved to {kg_npz}")

        # save metrics if available
        metrics = {}
        try:
            metrics_path = output_dir / f"{pipeline_model_arg}_metrics.json"
            metrics["mrr"] = float(result.metric_results.get_metric('both.realistic.inverse_harmonic_mean_rank') or 0.0)
            metrics["hits@1"] = float(result.metric_results.get_metric('both.realistic.hits_at_1') or 0.0)
            metrics["hits@3"] = float(result.metric_results.get_metric('both.realistic.hits_at_3') or 0.0)
            metrics["hits@10"] = float(result.metric_results.get_metric('both.realistic.hits_at_10') or 0.0)
        except Exception:
            metrics_path = output_dir / f"{pipeline_model_arg}_metrics.json"
            metrics = {"warning": "could not extract metrics"}
        with open(metrics_path, 'w') as fh:
            json.dump({"run_id": run_id, "model": pipeline_model_arg, "metrics": metrics, "train_time_s": train_time}, fh, indent=2)

        # cleanup
        del result
        del tf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return kg_npz

    except Exception as e:
        logger.exception(f"Run {run_id}: error saving embeddings: {e}")
        raise

# ---------- batch wrapper (per-run independent) ----------
def train_kg_embeddings_for_runs(
    base_data_dir: str = "scratch/project/path_planning/global_training_data",
    num_runs: int = 150,
    model_name: str = "ComplEx",
    embedding_dim: Optional[int] = None,
    num_epochs: int = 100,
    random_seed: int = 42,
    batch_size: int = 128,
    create_joint: bool = False  # explicit: we removed joint creation; kept arg for compatibility
) -> Path:
    """
    Iterate runs 1..num_runs, train each run independently if run_{i}/triples.tsv exists.
    Save per-run outputs to run_{i}/kg/
    Returns master output dir (base/kg_embeddings_summary)
    """
    pipeline_model_arg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["ComplEx"])["model"]
    base = Path(base_data_dir)
    master_output_dir = base / "kg_embeddings_summary"
    master_output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    successful_runs = 0
    failed_runs = []

    pbar = tqdm(total=num_runs, desc="Runs")
    for i in range(num_runs):
        run_id = i + 1
        run_dir = base / f"run_{run_id}"
        triples_path = run_dir / "triples.tsv"
        if not triples_path.exists():
            logger.warning(f"Run {run_id}: triples missing -> skip ({triples_path})")
            failed_runs.append(run_id)
            pbar.update(1)
            continue

        run_output_dir = run_dir / "kg"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            kg_npz = train_single_run_kg(
                data_path=str(triples_path),
                output_dir=str(run_output_dir),
                model_name=model_name,
                embedding_dim=embedding_dim,
                num_epochs=num_epochs,
                random_seed=random_seed,
                batch_size=batch_size
            )
            kg_npz_str = str(kg_npz)
            # read metrics file saved inside run_output_dir
            metrics_file = run_output_dir / f"{pipeline_model_arg}_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as fh:
                        m = json.load(fh)
                        m["run_id"] = run_id
                        m["kg_npz"] = kg_npz_str 
                        all_metrics.append(m)
                except Exception:
                    logger.warning(f"Run {run_id}: metric read failed")
            successful_runs += 1
        except Exception as e:
            logger.error(f"Run {run_id}: failed: {e}")
            failed_runs.append(run_id)
        finally:
            pbar.update(1)

    pbar.close()

    summary = {
        "total_runs": num_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "model": pipeline_model_arg,
        "embedding_dim": embedding_dim,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "random_seed": random_seed,
        "completed_at": pd.Timestamp.now().isoformat(),
    }
    if all_metrics:
        summary["runs"] = all_metrics
    summary_path = master_output_dir / "training_summary.json"
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    return master_output_dir
def canon_node_entity(node_label: str) -> str:
    # node_label 可能是 "9" 或 9 或 "node_9"
    s = str(node_label).strip()
    if s.startswith("node_"):
        return s
    if s.isdigit():
        return f"node_{int(s)}"
    # 兜底：如果你还有 node9 这种，就再补一条
    if s.startswith("node") and s[4:].isdigit():
        return f"node_{int(s[4:])}"
    return s

# ---------- example of how downstream code (GAT) can call load_run_kg_embeddings ----------
def example_load_for_gat(run_id: int, base_dir: str = "scratch/project/path_planning/global_training_data"):
    ent_emb, mapping = load_run_kg_embeddings(run_id, base_dir)
    if ent_emb is None:
        logger.info(f"Run {run_id}: no KG embeddings loaded")
        return None, None

    idx_file = Path(base_dir) / f"run_{run_id}" / "node_index_map.json"
    if not idx_file.exists():
        logger.warning(f"Run {run_id}: node_index_map.json missing; returning raw entity embeddings")
        return ent_emb, mapping

    with open(idx_file, "r") as f:
        node_index_map = json.load(f)

    n_nodes = len(node_index_map)
    emb_dim = ent_emb.shape[1]
    ordered = np.zeros((n_nodes, emb_dim), dtype=float)

    hits = 0
    for node_label, idx in node_index_map.items():
        key = canon_node_entity(node_label)

        ent_idx = None
        if key in mapping:
            ent_idx = mapping[key]
        else:
            pref = f"run_{run_id}_{key}"
            if pref in mapping:
                ent_idx = mapping[pref]

        if ent_idx is None:
            continue

        ii = int(idx)
        ei = int(ent_idx)
        if 0 <= ii < n_nodes and 0 <= ei < ent_emb.shape[0]:
            ordered[ii] = ent_emb[ei]
            hits += 1

    logger.info(f"Run {run_id}: KG hit {hits}/{n_nodes} ({hits/max(1,n_nodes):.1%})")
    return ordered, node_index_map

# ---------- main ----------
if __name__ == "__main__":
    # Example config - modify as needed
    config = {
        "base_data_dir": "scratch/project/path_planning/global_training_data",
        "num_runs": 13,
        "model_name": "TransE",
        "embedding_dim": 16,
        "num_epochs": 200,
        "random_seed": 42,
        "batch_size": 128,
        "create_joint": False
    }

    start = time.time()
    logger.info("Starting batch per-run KG training")
    master_dir = train_kg_embeddings_for_runs(
        base_data_dir=config["base_data_dir"],
        num_runs=config["num_runs"],
        model_name=config["model_name"],
        embedding_dim=config["embedding_dim"],
        num_epochs=config["num_epochs"],
        random_seed=config["random_seed"],
        batch_size=config["batch_size"],
        create_joint=config["create_joint"]
    )
    elapsed = time.time() - start
    logger.info(f"Batch finished in {elapsed:.2f}s. Summary saved to {master_dir}")
