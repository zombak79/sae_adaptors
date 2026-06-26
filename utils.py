from pathlib import Path
import numpy as np
import pandas as pd
import torch

from compresso import TopKSAETrainer, TopKSAEConfig, L1Normalize
from compresso.io import save_srp_tensor

from compresso_recsys.checkpoint import read_checkpoint, load_recsys_split
from compresso_recsys.retrieval import evaluate_item_embeddings_with_holdout
from compresso_recsys.checkpoint import save_json, load_json, update_checkpoint, update_stage_manifest

from sentence_transformers import SentenceTransformer

from dae import DAETrainer, DAEConfig
from vae import BetaVAETrainer, BetaVAEConfig

CHECKPOINT_PATH_PREFIX = "artifacts"

AMAZON_CONFIG = {
            "--split_mode": "item_split",
            "--metadata_text_fields": "title,features,description,categories",
            "--min_entity_text_words": "20",
            "--min_user_support": "10",
            "--item_min_support": "10",
            "--min_value_to_keep": "1.0",
            "--set_all_values_to": "1.0",
            "--min_source_items": "1",
            "--min_target_items": "1",
            "--annotation_source": "none",
}


CONFIG = {
    "GoodBooks10k": {
        "dataset": "goodbooks",
        "--split_mode": "item_split",
        "--min_user_support": "5",
        "--item_min_support": "1",
        "--min_value_to_keep": "4.0",
        "--set_all_values_to": "1.0",
        "--min_entity_text_words": "20",
    },
    "MovieLens20M": {
        "dataset": "ml20m",
        "--split_mode": "item_split",
        "--min_user_support": "5",
        "--item_min_support": "1",
        "--min_value_to_keep": "4.0",
        "--set_all_values_to": "1.0",
        "--min_entity_text_words": "20",
    },
    "Books": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Books",
    },
    "Electronics": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Electronics",
    },
    "Toys_and_Games": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Toys_and_Games",
    },
    "Video_Games": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Video_Games",
    },
    "Automotive": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Automotive",
    },
    "Baby_Products": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Baby_Products",
    },
    "Beauty_and_Personal_Care": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Beauty_and_Personal_Care",
    },
    
    "Clothing_Shoes_and_Jewelry": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Clothing_Shoes_and_Jewelry",
    },
    "Grocery_and_Gourmet_Food": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Grocery_and_Gourmet_Food",
    },
    "Health_and_Household": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Health_and_Household",
    },
    "Office_Products": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Office_Products",
    },
    "Sports_and_Outdoors": AMAZON_CONFIG | {
        "dataset": "amazon2023",
        "amazon_category": "Sports_and_Outdoors",
    },
}

sae_config = {
    "epochs": [50,100,200],
    "decay": [True, False],
    "lr": [1e-3, 5e-3, 1e-2],
    "hidden_dim" : 8192,
    "k": 128,
    "batch_size": 1024,
    "post_sparsify": L1Normalize(),
    "sparsify_score_mode": "abs",
    "sparsify_ste_alpha": 0.01,
    "device": "cuda:0",
}

dae_config = {
    "epochs": [50,100,200],
    "decay": [True, False],
    "lr": [1e-3, 5e-3, 1e-2],
    "latent_dim" : 256,
    "device": "cuda:0",
}

#latent_dim=256, beta_kl=1e-6,epochs=200, decay=True, lr=1e-3
vae_config = {
    "epochs": [50,100,200],
    "decay": [True, False],
    "lr": [1e-3, 5e-3, 1e-2],
    "latent_dim" : 256,
    "beta_kl": 1e-6,
    "device": "cuda:0",
}

def eval_three_metrics(item_embeddings, source_indices, target_indices, batch_size=1024, show_progress=True):
    out = {}
    for k in (20, 50, 100):
        metrics = evaluate_item_embeddings_with_holdout(
            item_embeddings=item_embeddings,
            source_indices=source_indices,
            target_indices=target_indices,
            k=k,
            score_batch_size=batch_size,
            show_progress=show_progress,
        )
        out.update({kk: vv for kk, vv in metrics.items() if kk != "n_eval_users"})
    return out

def build_checkpoints(config):
    import sys
    from compresso_recsys.scripts.build_checkpoint import main

    checkpoints = {}
    for k, v in config.items():
        params = ["compresso-recsys-build-checkpoint"]
        if "dataset" in v:
            if v["dataset"]=="amazon2023":
                params+=["--dataset", "amazon2023", "--amazon_category", v["amazon_category"]]
            else:
                params+=["--dataset", v["dataset"]]
            
            for x,y in v.items():
                if x[:2] == "--":
                    params+=[x,y]
    
            checkpoint = Path(f"{CHECKPOINT_PATH_PREFIX}/{k}.zip")
            params+=["--checkpoint_path", str(checkpoint)]
    
            if not checkpoint.exists():
                try:
                    print(f"Building checkpoint for {k}.")
                    old_argv = sys.argv
                    sys.argv = params
                    main()
                    checkpoints[k]={"path":checkpoint}
                except ValueError:
                    sys.argv = old_argv
                    print(f"****** ValueError for {k} ******")
                finally:
                    sys.argv = old_argv
            else:
                print(f"Skipping building checkpoint for {k}.")
                checkpoints[k]={"path":checkpoint}
    return checkpoints

def gather_stats(checkpoints):
    for k,c in checkpoints.items():
        with read_checkpoint(c["path"]) as root:
            print(f"Reading checkpoint {k}.")
            split = load_recsys_split(root)
            c["n_users"], c["n_items"] = split["x_train"].shape
            c["n_interactions"] = split["x_train"].nnz
            c["meta_sample"] = split["entity_metadata"][["item_id","entity_text"]].sample(5)            
    return pd.DataFrame([[k,v["n_users"], v["n_items"], v["n_interactions"]] for k,v in checkpoints.items()], columns=["Dataset", "Users", "Items", "Interactions"])


def gather_results(checkpoints):
    rows = []

    def _stage_model(stage_name):
        if stage_name.startswith("sbert"):
            return "sbert"
        if stage_name.startswith("sae:"):
            return "sae"
        if stage_name.startswith("dae:"):
            return "dae"
        if stage_name.startswith("vae:"):
            return "vae"
        return stage_name.split(":", 1)[0]

    def _stage_sbert_name(stage_name, stage_info):
        if stage_name.startswith("sbert"):
            return stage_info.get("model_name", stage_name.replace("sbert:", ""))
        base_stage = stage_info.get("base_stage", "")
        if base_stage.startswith("sbert"):
            return base_stage.replace("sbert:", "")
        return base_stage

    def _flatten_metrics(metrics):
        if not isinstance(metrics, dict):
            return {}
        return {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}

    def _metrics_from_stage_file(root, stage_name, model):
        path = root / stage_name / "metrics.json"
        if not path.exists():
            return {}

        stage_metrics = load_json(root, f"{stage_name}/metrics.json")
        metrics = {}

        # Current stages store their evaluation under model-specific keys,
        # while some baseline stages use "test_metrics".
        metric_keys = [
            "test_metrics",
            "sae_metrics",
            "dae_metrics",
            "vae_metrics",
            f"{model}_metrics",
            "metrics",
        ]
        for key in metric_keys:
            metrics.update(_flatten_metrics(stage_metrics.get(key)))

        # Keep this permissive so future flat numeric metrics are picked up too.
        metrics.update(_flatten_metrics(stage_metrics))
        return metrics

    for dataset, checkpoint in checkpoints.items():
        path = checkpoint["path"] if isinstance(checkpoint, dict) else checkpoint
        with read_checkpoint(path) as root:
            manifest = load_json(root, "manifest.json")

            for stage_name, stage_info in manifest.get("stages", {}).items():
                model = _stage_model(stage_name)
                if model not in {"sbert", "sae", "dae", "vae"}:
                    continue

                metrics = _metrics_from_stage_file(root, stage_name, model)
                metrics.update(_flatten_metrics(stage_info.get("metrics")))

                row = {
                    "dataset": dataset,
                    "sbert_name": _stage_sbert_name(stage_name, stage_info),
                    "model": model,
                    "stage": stage_name,
                    "base_stage": stage_info.get("base_stage", None),
                }
                row.update(metrics)
                rows.append(row)

    return pd.DataFrame(rows)


def train_and_eval_sbert(checkpoint, sbert_name, sbert=None):
    SBERT_DIR = f"sbert:{sbert_name.replace('/', '_')}"
    with read_checkpoint(checkpoint["path"]) as root:
        if (root / SBERT_DIR).is_dir():
            print("Sbert dir already exists, skipping.")
            metrics = load_json(root, f"{SBERT_DIR}/metrics.json")
            test_metrics = metrics["test_metrics"]
            return test_metrics
        split = load_recsys_split(root)
        item_ids = split["item_ids"]
        meta = split["entity_metadata"]
        x_train = split["x_train"]
        train_item_indices = split["train_item_indices"]
        val_item_indices = split["val_item_indices"]
        test_item_indices = split["test_item_indices"]
        val_source_indices = split["val_source_indices"]
        val_target_indices = split["val_target_indices"]
        test_source_indices = split["test_source_indices"]
        test_target_indices = split["test_target_indices"]

    if sbert is None:
        model = SentenceTransformer(sbert_name, device="cuda")
    else:
        model = sbert
        
    texts = meta["entity_text"].fillna("").astype(str).tolist()
    
    item_embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    test_metrics = eval_three_metrics(
        item_embeddings,
        test_source_indices,
        test_target_indices,
        batch_size=1024,
        show_progress=True,
    )

    with update_checkpoint(checkpoint["path"]) as root:
        stage_dir = root / SBERT_DIR
        stage_dir.mkdir(parents=True, exist_ok=True)
    
        np.save(stage_dir / "item_embeddings.npy", item_embeddings.astype("float32"))
    
        save_json(
            root,
            f"{SBERT_DIR}/metrics.json",
            {
                "test_metrics": test_metrics,
                "model_name": sbert_name,
                "text_columns": ["entity_text"],
                "normalize_embeddings": True,
                "embedding_dim": int(item_embeddings.shape[1]),
            },
        )
    
        update_stage_manifest(
            root,
            SBERT_DIR,
            {
                "model_name": sbert_name,
                "text_columns": ["entity_text"],
                "normalize_embeddings": True,
                "embedding_dim": int(item_embeddings.shape[1]),
                "metrics": test_metrics,
            },
        )

    
    return test_metrics
    

def build_grid(config):
    grid_params = {}
    params = {}
    for k in config.keys():
        if isinstance(config[k], list):
            grid_params[k] = config[k]
        else:
            params[k] = config[k]
    df = pd.DataFrame([params])
    for key, values in grid_params.items():
        df = df.merge(pd.DataFrame({key: values}), how="cross")
    return [df.iloc[i].to_dict()|params for i in range(len(df))]


def train_and_eval_sae(checkpoint, grid):
    with read_checkpoint(checkpoint) as root:    
        split = load_recsys_split(root)
    
        item_ids = split["item_ids"]
        meta = split["entity_metadata"]
    
        x_train = split["x_train"]
    
        train_item_indices = split["train_item_indices"]
        val_item_indices = split["val_item_indices"]
        test_item_indices = split["test_item_indices"]
    
        val_source_indices = split["val_source_indices"]
        val_target_indices = split["val_target_indices"]
        test_source_indices = split["test_source_indices"]
        test_target_indices = split["test_target_indices"]
    
        manifest = load_json(root, "manifest.json")
    
    sbert_dirs = [x for x in manifest['stages'].keys() if x[:5] == 'sbert']
    
    for sbert_dir in sbert_dirs:
        print(f"Evaluating {sbert_dir}.")
        sae_dir = f"sae:{sbert_dir}"
        
        with read_checkpoint(checkpoint) as root:
            if (root / sae_dir).is_dir():
                print(f"SAE dir {sae_dir} already exists, skipping.")
                do_loop = False
            else:
                item_embeddings = np.load(root / sbert_dir / "item_embeddings.npy")
                print(f"Training and evaluating SAE dir {sae_dir}.")
                do_loop = True
        
        if do_loop:
            res={}
        
            for i in range(len(grid)):
                conf = grid[i]
                
                print(f"Config {i+1} out of {len(grid)}")
                print(conf)
                sae = TopKSAETrainer(
                    TopKSAEConfig(
                        **conf
                    )
                )
                
                # Important: fit only on train items.
                sae.fit(item_embeddings[train_item_indices])
                
                # Transform all items, including cold target items.
                sparse_srp = sae.transform(item_embeddings)
                metrics = evaluate_item_embeddings_with_holdout(
                        item_embeddings=sparse_srp.to_dense().detach().numpy(),
                        source_indices=val_source_indices[:10000],
                        target_indices=val_target_indices[:10000],
                        k=100,
                        score_batch_size=1024,
                        show_progress=True,
                    )
                
                res[i] = metrics["ndcg@100"]
                print(f"ndcg@100 = {res[i]}")    
        
            winning_conf = grid[max(res, key=res.get)]
            sae = TopKSAETrainer(
                TopKSAEConfig(
                    **winning_conf
                )
            )
            
            # Important: fit only on train items.
            sae.fit(item_embeddings[train_item_indices])
            
            # Transform all items, including cold target items.
            sparse_srp = sae.transform(item_embeddings)
        
            metrics = eval_three_metrics(
                sparse_srp.to_dense().detach().numpy(),
                test_source_indices,
                test_target_indices,
                batch_size=1024,
                show_progress=True,
            )
            
            print("Test metrics:")
            print(metrics)
            
            sae_dir = f"sae:{sbert_dir}"
            
            with update_checkpoint(checkpoint) as root:
                stage_dir = root / sae_dir
                stage_dir.mkdir(parents=True, exist_ok=True)
            
                # Save trained SAE model.
                torch.save(
                    {
                        "model_state_dict": sae.sae.state_dict(),
                        "config": sae.cfg.__dict__,
                        "input_dim": int(sae.sae.input_dim),
                        "hidden_dim": int(sae.cfg.hidden_dim),
                        "k": int(sae.cfg.k),
                        "score_mode": sae.cfg.sparsify_score_mode,
                        "ste_alpha": float(sae.cfg.sparsify_ste_alpha),
                        "post_sparsify": (
                            type(sae.cfg.post_sparsify).__name__
                            if sae.cfg.post_sparsify is not None
                            else None
                        ),
                    },
                    stage_dir / "model.pt",
                )
            
                save_srp_tensor(stage_dir / "sparse_embeddings.srp.pt", sparse_srp)
            
                save_json(
                    root,
                    f"{sae_dir}/metrics.json",
                    {
                        "sae_metrics": metrics,
                        "base_stage": sbert_dir,
                        "hidden_dim": int(sae.cfg.hidden_dim),
                        "k": int(sae.cfg.k),
                        "score_mode": sae.cfg.sparsify_score_mode,
                        "ste_alpha": float(sae.cfg.sparsify_ste_alpha),
                        "post_sparsify": (
                            type(sae.cfg.post_sparsify).__name__
                            if sae.cfg.post_sparsify is not None
                            else None
                        ),
                    },
                )
            
                update_stage_manifest(
                    root,
                    sae_dir,
                    {
                        "model_path": f"{sae_dir}/model.pt",
                        "sparse_embeddings_path": f"{sae_dir}/sparse_embeddings.srp.pt",
                        "base_stage": sbert_dir,
                        "hidden_dim": int(sae.cfg.hidden_dim),
                        "k": int(sae.cfg.k),
                        "score_mode": sae.cfg.sparsify_score_mode,
                        "ste_alpha": float(sae.cfg.sparsify_ste_alpha),
                        "metrics": metrics,
                    },
                )
            
                     
            
                
        
def train_and_eval_dae(checkpoint, grid):
    with read_checkpoint(checkpoint) as root:
        split = load_recsys_split(root)

        train_item_indices = split["train_item_indices"]

        val_source_indices = split["val_source_indices"]
        val_target_indices = split["val_target_indices"]
        test_source_indices = split["test_source_indices"]
        test_target_indices = split["test_target_indices"]

        manifest = load_json(root, "manifest.json")

    sbert_dirs = [x for x in manifest["stages"].keys() if x.startswith("sbert")]

    for sbert_dir in sbert_dirs:
        print(f"Evaluating {sbert_dir}.")
        dae_dir = f"dae:{sbert_dir}"

        with read_checkpoint(checkpoint) as root:
            if (root / dae_dir).is_dir():
                print(f"DAE dir {dae_dir} already exists, skipping.")
                do_loop = False
            else:
                item_embeddings = np.load(root / sbert_dir / "item_embeddings.npy").astype("float32")
                print(f"Training and evaluating DAE dir {dae_dir}.")
                do_loop = True

        if not do_loop:
            continue

        res = {}

        for i, conf in enumerate(grid):
            print(f"Config {i + 1} out of {len(grid)}")
            print(conf)

            cfg = DAEConfig(**conf)

            dae = DAETrainer(cfg)

            # Important: fit only on train items.
            dae.fit(item_embeddings[train_item_indices])
            
            # Transform all items, including cold target items.
            dae_embeddings = dae.transform(item_embeddings).cpu().detach().numpy()
            
            metrics = evaluate_item_embeddings_with_holdout(
                item_embeddings=dae_embeddings,
                source_indices=val_source_indices[:10000],
                target_indices=val_target_indices[:10000],
                k=100,
                score_batch_size=1024,
                show_progress=True,
            )

            res[i] = metrics["ndcg@100"]
            print(f"ndcg@100 = {res[i]}")

        winning_conf = grid[max(res, key=res.get)]
        print("Winning config:")
        print(winning_conf)

        cfg = DAEConfig(**winning_conf)

        dae = DAETrainer(cfg)

        # Important: fit only on train items.
        dae.fit(item_embeddings[train_item_indices])
        
        # Transform all items, including cold target items.
        dae_embeddings = dae.transform(item_embeddings).cpu().detach().numpy()

        metrics = eval_three_metrics(
            dae_embeddings,
            test_source_indices,
            test_target_indices,
            batch_size=1024,
            show_progress=True,
        )

        print("Test metrics:")
        print(metrics)

        with update_checkpoint(checkpoint) as root:
            stage_dir = root / dae_dir
            stage_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": dae.state_dict(),
                    "config": cfg.__dict__,
                    "base_stage": sbert_dir,
                    "output_stage": dae_dir,
                },
                stage_dir / "model.pt",
            )

            np.save(stage_dir / "item_embeddings.npy", dae_embeddings.astype("float32"))

            save_json(
                root,
                f"{dae_dir}/metrics.json",
                {
                    "dae_metrics": metrics,
                    "base_stage": sbert_dir,
                    "winning_config": winning_conf,
                    "grid_results": {str(k): float(v) for k, v in res.items()},
                    "embedding_dim": int(dae_embeddings.shape[1]),
                },
            )

            update_stage_manifest(
                root,
                dae_dir,
                {
                    "model_path": f"{dae_dir}/model.pt",
                    "item_embeddings_path": f"{dae_dir}/item_embeddings.npy",
                    "base_stage": sbert_dir,
                    "winning_config": winning_conf,
                    "metrics": metrics,
                    "embedding_dim": int(dae_embeddings.shape[1]),
                },
            )


def train_and_eval_vae(checkpoint, grid):
    with read_checkpoint(checkpoint) as root:
        split = load_recsys_split(root)

        train_item_indices = split["train_item_indices"]

        val_source_indices = split["val_source_indices"]
        val_target_indices = split["val_target_indices"]
        test_source_indices = split["test_source_indices"]
        test_target_indices = split["test_target_indices"]

        manifest = load_json(root, "manifest.json")

    sbert_dirs = [x for x in manifest["stages"].keys() if x.startswith("sbert")]

    for sbert_dir in sbert_dirs:
        print(f"Evaluating {sbert_dir}.")
        vae_dir = f"vae:{sbert_dir}"

        with read_checkpoint(checkpoint) as root:
            if (root / vae_dir).is_dir():
                print(f"VAE dir {vae_dir} already exists, skipping.")
                do_loop = False
            else:
                item_embeddings = np.load(root / sbert_dir / "item_embeddings.npy").astype("float32")
                print(f"Training and evaluating VAE dir {vae_dir}.")
                do_loop = True

        if not do_loop:
            continue

        res = {}

        for i, conf in enumerate(grid):
            print(f"Config {i + 1} out of {len(grid)}")
            print(conf)

            cfg = BetaVAEConfig(**conf)

            vae = BetaVAETrainer(cfg)

            # Important: fit only on train items.
            vae.fit(item_embeddings[train_item_indices])

            # Transform all items, including cold target items.
            vae_embeddings = vae.transform(item_embeddings).cpu().detach().numpy()

            metrics = evaluate_item_embeddings_with_holdout(
                item_embeddings=vae_embeddings,
                source_indices=val_source_indices[:10000],
                target_indices=val_target_indices[:10000],
                k=100,
                score_batch_size=1024,
                show_progress=True,
            )

            res[i] = metrics["ndcg@100"]
            print(f"ndcg@100 = {res[i]}")

        winning_conf = grid[max(res, key=res.get)]
        print("Winning config:")
        print(winning_conf)

        cfg = BetaVAEConfig(**winning_conf)

        vae = BetaVAETrainer(cfg)

        # Important: fit only on train items.
        vae.fit(item_embeddings[train_item_indices])

        # Transform all items, including cold target items.
        vae_embeddings = vae.transform(item_embeddings).cpu().detach().numpy()

        metrics = eval_three_metrics(
            vae_embeddings,
            test_source_indices,
            test_target_indices,
            batch_size=1024,
            show_progress=True,
        )

        print("Test metrics:")
        print(metrics)

        with update_checkpoint(checkpoint) as root:
            stage_dir = root / vae_dir
            stage_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": vae.state_dict(),
                    "config": cfg.__dict__,
                    "base_stage": sbert_dir,
                    "output_stage": vae_dir,
                },
                stage_dir / "model.pt",
            )

            np.save(stage_dir / "item_embeddings.npy", vae_embeddings.astype("float32"))

            save_json(
                root,
                f"{vae_dir}/metrics.json",
                {
                    "vae_metrics": metrics,
                    "base_stage": sbert_dir,
                    "winning_config": winning_conf,
                    "grid_results": {str(k): float(v) for k, v in res.items()},
                    "embedding_dim": int(vae_embeddings.shape[1]),
                },
            )

            update_stage_manifest(
                root,
                vae_dir,
                {
                    "model_path": f"{vae_dir}/model.pt",
                    "item_embeddings_path": f"{vae_dir}/item_embeddings.npy",
                    "base_stage": sbert_dir,
                    "winning_config": winning_conf,
                    "metrics": metrics,
                    "embedding_dim": int(vae_embeddings.shape[1]),
                },
            )
