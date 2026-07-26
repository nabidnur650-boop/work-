from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .data import build_loaders
from .metrics import (
    chunk_iou,
    evidence_recall_at_k,
    logits_to_intervals,
    sequence_exact_match,
    token_accuracy,
)
from .train import _to_device, build_model
from .utils import load_config, resolve_device, save_json, set_seed


@torch.no_grad()
def evaluate(config_path: str, checkpoint_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    train_cfg = config.get("training", {})
    torch.set_num_threads(int(train_cfg.get("num_threads", 4)))
    device = resolve_device(str(train_cfg.get("device", "auto")))
    _, _, test_loader = build_loaders(config["data"], seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint.get("config", config), audio_dim=int(checkpoint["audio_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

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
    examples: list[dict[str, Any]] = []
    gate_values: list[torch.Tensor] = []
    surprise_values: list[torch.Tensor] = []

    for batch in test_loader:
        raw_ids = batch["example_id"]
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

        diagnostics = output["memory_diagnostics"]
        if diagnostics is not None:
            gate_values.append(diagnostics.gates.detach().cpu())
            surprise_values.append(diagnostics.surprises.detach().cpu())

        if len(examples) < 8:
            predicted = output["decoder_logits"].argmax(dim=-1).detach().cpu()
            intervals = logits_to_intervals(
                output["evidence_logits"].detach().cpu(),
                batch["chunk_start_sec"].detach().cpu(),
                batch["chunk_end_sec"].detach().cpu(),
                batch["audio_attention_mask"].detach().cpu(),
                top_k=model.evidence_top_k,
            )
            for row in range(min(batch_size, 8 - len(examples))):
                examples.append(
                    {
                        "example_id": str(raw_ids[row]),
                        "predicted_ids": predicted[row].tolist(),
                        "label_ids": batch["labels"][row].detach().cpu().tolist(),
                        "top_evidence_intervals": intervals[row],
                        "gold_evidence_chunks": torch.nonzero(
                            batch["evidence_targets"][row] > 0,
                            as_tuple=False,
                        ).flatten().detach().cpu().tolist(),
                    }
                )

    result: dict[str, Any] = {key: value / max(count, 1) for key, value in totals.items()}
    result.update(
        {
            "n_examples": count,
            "checkpoint": checkpoint_path,
            "evidence_top_k": model.evidence_top_k,
            "sample_predictions": examples,
        }
    )
    if gate_values:
        result["mean_memory_gate"] = float(torch.cat(gate_values).mean())
        result["mean_memory_surprise"] = float(torch.cat(surprise_values).mean())

    output_dir = Path(config.get("output_dir", "results/run"))
    save_json(result, output_dir / "test_metrics.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    result = evaluate(args.config, args.checkpoint)
    for key, value in result.items():
        if key != "sample_predictions":
            print(f"{key}: {value}")
    print(f"sample_predictions: {len(result['sample_predictions'])} written to test_metrics.json")


if __name__ == "__main__":
    main()
