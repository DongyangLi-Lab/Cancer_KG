"""Explicit external-source/weight binding for the complete native frozen TXEncoder."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch
from cancer_kg.encoders.native import load_tahoe_local_audited


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder(config):
    if config["device"] != "cuda":
        raise ValueError("The registered complete Tahoe path uses CUDA")
    upstream = json.loads((Path(__file__).resolve().parents[3] / "docs" / "UPSTREAM.json").read_text())
    for relative, digest in upstream["source_files_sha256"].items():
        if sha(Path(config["repo_dir"]) / relative) != digest:
            raise ValueError("Pinned native source differs")
    for name in ("best-model.pt", "model_config.yml", "collator_config.yml", "vocab.json"):
        if sha(Path(config["model_dir"]) / name) != config["asset_sha256"][name]:
            raise ValueError("Trusted official Tahoe asset differs")
    sys.path.insert(0, config["repo_dir"])
    native, vocab, model_config, collator, audit = load_tahoe_local_audited(config)
    encoder = native.transformer_encoder
    if (int(model_config["d_model"]) != 2560 or int(model_config["n_layers"]) != 32
            or int(model_config["n_heads"]) != 20 or len(encoder.layers) != 32
            or encoder.use_attn_mask or model_config["attn_config"]["attn_impl"] != "flash"):
        raise ValueError("Requires the complete pretrained 32-layer 2560-width 20-head encoder")
    for block in encoder.layers:
        attention = block.self_attn
        if attention.attn_impl != "flash" or "flash" not in attention.attn_fn.__name__.lower():
            raise ValueError("The native Flash route differs")
    encoder.eval().requires_grad_(False)
    audit.update(layers=32, width=2560, heads=20, source_revision=upstream["revision"],
                 pretrained_encoder_parameters=sum(p.numel() for p in encoder.parameters()),
                 asset_sha256=config["asset_sha256"])
    del native, vocab, collator
    gc.collect()
    torch.cuda.empty_cache()
    return encoder, audit
