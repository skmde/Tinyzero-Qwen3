import argparse
import json
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU preflight checks for TinyZero Qwen RL setup")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    args = parser.parse_args()

    print("[1/5] Import checks...")
    import torch
    import pandas as pd
    from transformers import AutoConfig, AutoTokenizer

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")

    print("[2/5] Path checks...")
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not os.path.isfile(args.train_file):
        raise FileNotFoundError(f"Train parquet not found: {args.train_file}")
    if not os.path.isfile(args.val_file):
        raise FileNotFoundError(f"Val parquet not found: {args.val_file}")

    print("[3/5] Tokenizer/config checks...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    cfg_model_type = None
    try:
        cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        cfg_model_type = cfg.model_type
    except Exception as exc:
        # Older transformers in TinyZero may not recognize newer model_type (e.g., qwen3).
        config_path = os.path.join(args.model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        cfg_model_type = raw_cfg.get("model_type", "unknown")
        print(f"AutoConfig fallback due to: {exc}")

    print(f"tokenizer_class={tok.__class__.__name__}")
    print(f"model_type={cfg_model_type}")

    print("[4/5] Parquet readability checks...")
    train_df = pd.read_parquet(args.train_file)
    val_df = pd.read_parquet(args.val_file)
    print(f"train_rows={len(train_df)}")
    print(f"val_rows={len(val_df)}")

    print("[5/5] Basic schema checks...")
    print(f"train_columns={list(train_df.columns)}")
    print(f"val_columns={list(val_df.columns)}")

    print("CPU preflight passed. You can move to GPU cloud training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
