from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm import tqdm

from .data import build_loaders
from .metrics import chunk_iou, evidence_recall_at_k, sequence_exact_match, token_accuracy
from .model import EviAudioMT
from .utils import count_parameters, load_config, resolve_device, save_json, set_seed


def build_model(config: dict[str, Any], audio_dim: int) -> EviAudioMT:
    model_cfg = config["model"]
    decoder_cfg = model_cfg.get("decoder", {})
    memory_cfg = model_cfg.get("memory", {})
    train_cfg = config.get("training", {})
    return EviAudioMT(
        audio_dim=audio_dim,
        d_model=int(model_cfg.get("d_model", 256)),
        n_blocks=int(model_cfg.get("n_blocks", 4)),
        backend=str(model_cfg.get("backend", "fallback")),
        d_state=int(model_cfg.get("d_state", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        evidence_top_k=int(model_cfg.get("evidence_top_k", 8)),
        decoder_type=str(decoder_cfg.get("type", "toy")),
        decoder_name=decoder_cfg.get("name"),
        decoder_revision=decoder_cfg.get("revision"),
        decoder_frozen=bool(decoder_cfg.get("frozen", False)),
        decoder_vocab_size=int(decoder_cfg.get("vocab_size", 128)),
        decoder_d_model=int(decoder_cfg.get("d_model", 128)),
        decoder_max_target_length=int(decoder_cfg.get("max_target_length", 32)),
        memory_enabled=bool(memory_cfg.get("enabled", True)),
        memory_decay=float(memory_cfg.get("decay", 0.97)),
        memory_update_rate=float(memory_cfg.get("update_rate", 0.05)),
        memory_max_state_norm=float(memory_cfg.get("max_state_norm", 50.0)),
        memory_detach_returned_state=bool(memory_cfg.get("detach_returned_state", True)),
        memory_gate_mode=str(memory_cfg.get("gate_mode", "learned")),
        memory_constant_gate=float(memory_cfg.get("constant_gate", 0.5)),
        control_adapter=str(model_cfg.get("control_adapter", "none")),
        stream_window_size=(
            int(model_cfg["stream_window_size"])
            if model_cfg.get("stream_window_size") is not None
            else None
        ),
        evidence_weight=float(train_cfg.get("evidence_weight", 1.0)),
        gate_regularization_weight=float(train_cfg.get("gate_regularization_weight", 0.0)),
        gate_target_density=float(train_cfg.get("gate_target_density", 0.15)),
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_loader(model: EviAudioMT, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "text_loss": 0.0,
        "evidence_loss": 0.0,
        "token_accuracy": 0.0,
        "sequence_exact_match": 0.0,
        "evidence_recall_at_k": 0.0,
        "chunk_iou": 0.0,
    }
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        output = model(
            batch["audio_embeddings"],
            batch["prompt_input_ids"],
            batch["prompt_attention_mask"],
            labels=batch["labels"],
            evidence_targets=batch["evidence_targets"],
            audio_attention_mask=batch["audio_attention_mask"],
        )
        batch_size = batch["audio_embeddings"].shape[0]
        count += batch_size
        totals["loss"] += float(output["loss"].item()) * batch_size
        totals["text_loss"] += float(output["text_loss"].item()) * batch_size
        totals["evidence_loss"] += float(output["evidence_loss"].item()) * batch_size
        totals["token_accuracy"] += token_accuracy(output["decoder_logits"], batch["labels"]) * batch_size
        totals["sequence_exact_match"] += sequence_exact_match(
            output["decoder_logits"], batch["labels"]
        ) * batch_size
        totals["evidence_recall_at_k"] += evidence_recall_at_k(
            output["evidence_logits"],
            batch["evidence_targets"],
            batch["audio_attention_mask"],
            k=model.evidence_top_k,
        ) * batch_size
        totals["chunk_iou"] += chunk_iou(
            output["evidence_logits"],
            batch["evidence_targets"],
            batch["audio_attention_mask"],
        ) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(config_path: str) -> Path:
    config = load_config(config_path)
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    train_cfg = config.get("training", {})
    torch.set_num_threads(int(train_cfg.get("num_threads", 4)))
    device = resolve_device(str(train_cfg.get("device", "auto")))
    train_loader, validation_loader, _ = build_loaders(config["data"], seed)

    first_batch = next(iter(train_loader))
    audio_dim = int(first_batch["audio_embeddings"].shape[-1])
    model = build_model(config, audio_dim=audio_dim).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg.get("epochs", 20))
    patience = int(train_cfg.get("patience", 5))
    gradient_clip = float(train_cfg.get("gradient_clip", 1.0))

    output_dir = Path(config.get("output_dir", "results/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")
    save_json(
        {
            "device": str(device),
            "parameters": count_parameters(model),
            "audio_dim": audio_dim,
            "evidence_top_k": model.evidence_top_k,
        },
        output_dir / "metadata.json",
    )

    best_loss = float("inf")
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["audio_embeddings"],
                batch["prompt_input_ids"],
                batch["prompt_attention_mask"],
                labels=batch["labels"],
                evidence_targets=batch["evidence_targets"],
                audio_attention_mask=batch["audio_attention_mask"],
            )
            loss = output["loss"]
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite or missing loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            batch_size = batch["audio_embeddings"].shape[0]
            running += float(loss.item()) * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(seen, 1)
        validation = evaluate_loader(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, **validation}
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_loss:.5f} val_loss={validation['loss']:.5f} "
            f"val_token_acc={validation['token_accuracy']:.3f} "
            f"val_evidence_R@K={validation['evidence_recall_at_k']:.3f}"
        )

        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "audio_dim": audio_dim,
                    "epoch": epoch,
                    "validation": validation,
                },
                output_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break
    save_json(history, output_dir / "history.json")
    return output_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    checkpoint = train(args.config)
    print(f"Saved best checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
