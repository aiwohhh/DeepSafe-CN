#!/usr/bin/env python3
"""
DeepSafe Ensemble Retraining Pipeline
======================================

One-command pipeline that:
  1. Health-checks all active model services
  2. Runs inference against the public dataset to generate meta-features
  3. Trains the meta-learner ensemble
  4. Deploys artifacts to api/meta_model_artifacts/

Usage:
  python scripts/retrain_pipeline.py --media-type image
  python scripts/retrain_pipeline.py --media-type image --dataset-dir ./public_dataset/images
  python scripts/retrain_pipeline.py --media-type image --skip-generate --meta-csv ./meta_learning_data/meta_features_image.csv
"""

import argparse
import json
import os
import re
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    sys.exit(1)

DATASET_DIR_MAP = {
    "image": os.path.join(PROJECT_ROOT, "public_dataset", "images"),
    "video": os.path.join(PROJECT_ROOT, "public_dataset", "video"),
    "audio": os.path.join(PROJECT_ROOT, "public_dataset", "audio"),
}

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "deepsafe_config.json")
META_DATA_DIR = os.path.join(PROJECT_ROOT, "meta_learning_data")
API_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "api", "meta_model_artifacts")
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "meta_learning_experiment_runs")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def extract_port(url):
    """Extract port from a URL like http://service_name:5001/predict."""
    match = re.search(r":(\d+)", url)
    return int(match.group(1)) if match else None


def health_check_models(config, media_type):
    """Check which models are healthy and return their names."""
    mt_config = config.get("media_types", {}).get(media_type, {})
    health_endpoints = mt_config.get("health_endpoints", {})

    if not health_endpoints:
        print(f"  No models configured for '{media_type}'")
        return []

    healthy = []
    unhealthy = []

    for model_name, health_url in health_endpoints.items():
        port = extract_port(health_url)
        local_url = f"http://localhost:{port}/health"
        try:
            r = requests.get(local_url, timeout=10)
            status = r.json().get("status", "unknown")
            if status in ("healthy", "not_loaded", "degraded_not_loaded"):
                healthy.append(model_name)
                print(f"  {model_name:<30} OK ({status})")
            else:
                unhealthy.append(model_name)
                print(f"  {model_name:<30} DEGRADED ({status})")
        except Exception as e:
            unhealthy.append(model_name)
            print(f"  {model_name:<30} UNREACHABLE ({e})")

    return healthy


def run_feature_generation(media_type, dataset_dir, output_csv, healthy_models):
    """Generate meta-features by querying all healthy models."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Generating meta-features")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output:  {output_csv}")
    print(f"  Models:  {', '.join(healthy_models)}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "meta_feature_generator.py"),
        "--media-type",
        media_type,
        "--input-dir",
        dataset_dir,
        "--output-csv",
        output_csv,
        "--config-path",
        CONFIG_PATH,
        "--specific-models",
        ",".join(healthy_models),
    ]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"ERROR: Meta-feature generation failed (exit code {result.returncode})")
        sys.exit(1)

    if not os.path.exists(output_csv):
        print(f"ERROR: Expected output CSV not found at {output_csv}")
        sys.exit(1)

    print(f"\nMeta-features saved to {output_csv}")


def run_training(media_type, meta_csv, optimizer, trials):
    """Train the meta-learner ensemble."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Training meta-learner ensemble")
    print(f"  Features: {meta_csv}")
    print(f"  Optimizer: {optimizer} ({trials} trials)")
    print(f"  Artifacts: {API_ARTIFACTS_DIR}/{media_type}/")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "train_meta_learner_advanced.py"),
        "--media-type",
        media_type,
        "--meta-file",
        meta_csv,
        "--output-dir",
        EXPERIMENT_DIR,
        "--api-artifacts-dir",
        API_ARTIFACTS_DIR,
        "--optimizer",
        optimizer,
        "--optuna-trials",
        str(trials),
    ]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"ERROR: Meta-learner training failed (exit code {result.returncode})")
        sys.exit(1)


def verify_artifacts(media_type):
    """Verify all required artifacts exist after training."""
    artifacts_dir = os.path.join(API_ARTIFACTS_DIR, media_type)
    required_files = [
        "deepsafe_meta_learner.joblib",
        "deepsafe_meta_scaler.joblib",
        "deepsafe_meta_imputer.joblib",
        "deepsafe_meta_feature_columns.json",
    ]

    print(f"\n{'='*60}")
    print(f"STEP 4: Verifying artifacts")
    print(f"{'='*60}")

    all_ok = True
    for fname in required_files:
        fpath = os.path.join(artifacts_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  [OK] {fname} ({size:,} bytes)")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False

    # Show feature columns
    cols_path = os.path.join(artifacts_dir, "deepsafe_meta_feature_columns.json")
    if os.path.exists(cols_path):
        with open(cols_path) as f:
            cols = json.load(f)
        print(f"\n  Feature columns ({len(cols)} models):")
        for col in cols:
            print(f"    - {col}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="DeepSafe one-command ensemble retraining pipeline"
    )
    parser.add_argument(
        "--media-type",
        required=True,
        choices=["image", "video", "audio"],
        help="Media type to retrain ensemble for",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to labeled dataset (default: public_dataset/{media_type})",
    )
    parser.add_argument(
        "--optimizer",
        default="gridsearch",
        choices=["optuna", "gridsearch"],
        help="Hyperparameter optimizer (default: gridsearch)",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip meta-feature generation (use existing CSV)",
    )
    parser.add_argument(
        "--meta-csv",
        default=None,
        help="Path to existing meta-features CSV (with --skip-generate)",
    )
    parser.add_argument(
        "--restart-api",
        action="store_true",
        help="Restart the API container after training to load new artifacts",
    )
    args = parser.parse_args()

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"DeepSafe Ensemble Retraining Pipeline")
    print(f"  Media type: {args.media_type}")
    print(f"{'='*60}")

    # Step 1: Health check
    print(f"\n{'='*60}")
    print(f"STEP 1: Health-checking model services")
    print(f"{'='*60}")

    config = load_config()
    healthy_models = health_check_models(config, args.media_type)

    if not healthy_models:
        print(
            f"\nERROR: No healthy models found for '{args.media_type}'. Start services first: make start"
        )
        sys.exit(1)

    print(f"\n  {len(healthy_models)} model(s) available for training")

    # Step 2: Generate meta-features (or skip)
    dataset_dir = args.dataset_dir or DATASET_DIR_MAP.get(args.media_type)
    os.makedirs(META_DATA_DIR, exist_ok=True)
    output_csv = os.path.join(META_DATA_DIR, f"meta_features_{args.media_type}.csv")

    if args.skip_generate:
        meta_csv = args.meta_csv or output_csv
        if not os.path.exists(meta_csv):
            print(f"ERROR: Meta-features CSV not found at {meta_csv}")
            sys.exit(1)
        print(f"\nSkipping feature generation. Using: {meta_csv}")
        output_csv = meta_csv
    else:
        if not dataset_dir or not os.path.exists(dataset_dir):
            print(f"ERROR: Dataset directory not found: {dataset_dir}")
            sys.exit(1)
        run_feature_generation(args.media_type, dataset_dir, output_csv, healthy_models)

    # Step 3: Train meta-learner
    run_training(args.media_type, output_csv, args.optimizer, args.optuna_trials)

    # Step 4: Verify
    artifacts_ok = verify_artifacts(args.media_type)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    if artifacts_ok:
        print(f"SUCCESS! Pipeline completed in {elapsed:.1f}s")
        print(f"\nArtifacts deployed to: {API_ARTIFACTS_DIR}/{args.media_type}/")
        if args.restart_api:
            print("\nRestarting API container...")
            subprocess.run(["docker", "compose", "restart", "api"], cwd=PROJECT_ROOT)
        else:
            print("\nRestart the API to load the new meta-learner:")
            print("  docker compose restart api")
    else:
        print(f"WARNING: Some artifacts missing. Check training output for errors.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
