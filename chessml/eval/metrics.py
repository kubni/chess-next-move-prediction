import numpy as np
import numpy.typing as npt
import torch

from chessml.encoding.schema import META_ELO_BUCKET, META_PLY
from chessml.eval.legality_mask import apply_mask, legal_mask

TOP_KS = (1, 3, 5)

# Ply cutoffs between game phases
PHASE_EDGES = (20, 60)
PHASE_NAMES = ("opening", "middlegame", "endgame")


def top_k_hits(logits: torch.Tensor, labels: torch.Tensor, k: int) -> npt.NDArray[np.bool]:
    """One boolean per position: was the played move among the k highest logits."""
    top = logits.topk(k, dim=1).indices          # (Batch pozicija, k)
    return (top == labels[:, None]).any(dim=1).numpy()


def evaluate(
    predict_fn,
    positions: npt.NDArray[np.int8],
    labels: npt.NDArray[np.int16],
    meta: npt.NDArray[np.int32],
    move_to_index: dict[str, int],
    batch_size: int = 1024,
    model_inputs: npt.NDArray | None = None,
) -> dict:
    """
    Score a model over a whole split.

    Args:
        predict_fn: positions batch (b, BOARD_DIM) -> logits (b, VOCAB_SIZE). A
            function rather than a model, because the flat models want to_planes
            and the transformer wants token sequences: different inputs, same
            output.
        model_inputs: what predict_fn is fed, row for row aligned with positions.
            Defaults to positions themselves. The transformer reads token windows
            while the mask still has to be built from the boards, so the two are
            sliced apart — but always over the same range of rows.

    Returns:
        Overall top-1/3/5 with and without the legality mask, the illegal-move
        rate, and the same figures split by Elo bucket and by game phase.
    """

    # model_inputs only used for transformers, because it needs tokens not 
    # positions.
    if model_inputs is not None and len(model_inputs) != len(positions):
        raise ValueError(
            f"model_inputs has {len(model_inputs)} rows, positions has {len(positions)}"
        )

    inputs = positions if model_inputs is None else model_inputs
    n = len(positions)
    hits = {k: np.zeros(n, dtype=np.bool) for k in TOP_KS}
    hits_masked = {k: np.zeros(n, dtype=np.bool) for k in TOP_KS}
    illegal = np.zeros(n, dtype=np.bool)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        positions_batch = positions[start:stop]
        y = torch.from_numpy(labels[start:stop].astype(np.int64))

        with torch.no_grad():
            logits = predict_fn(inputs[start:stop])

        mask = torch.from_numpy(legal_mask(positions_batch, move_to_index))
        masked = apply_mask(logits, mask)

        for k in TOP_KS:
            hits[k][start:stop] = top_k_hits(logits, y, k)
            hits_masked[k][start:stop] = top_k_hits(masked, y, k)

        # Measured on the raw prediction only: with the mask applied an illegal
        # move is impossible by construction, so it would always report zero.
        top1 = logits.argmax(dim=1)
        positions_local_to_batch = torch.arange(stop - start)  # (b,) positions 0..b-1, local to this batch
        picked_is_legal = mask[positions_local_to_batch, top1] # (b,) pairs positions i with column top1[i]
        illegal[start:stop] = ~picked_is_legal.numpy()

    def summarize(rows: npt.NDArray[np.bool]) -> dict:
        out = {f"top{k}": float(hits[k][rows].mean()) for k in TOP_KS}
        out |= {f"top{k}_masked": float(hits_masked[k][rows].mean()) for k in TOP_KS}
        out["illegal_rate"] = float(illegal[rows].mean())
        out["n"] = int(rows.sum())
        return out

    result = summarize(np.ones(n, dtype=np.bool))

    elo = meta[:, META_ELO_BUCKET]
    result["by_elo"] = {int(b): summarize(elo == b) for b in np.unique(elo)}

    phase = np.digitize(meta[:, META_PLY], PHASE_EDGES)
    result["by_phase"] = {PHASE_NAMES[p]: summarize(phase == p) for p in np.unique(phase)}

    return result