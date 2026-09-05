from collections.abc import Callable
from chessml.encoding.schema import LABEL_DTYPE, POSITION_DTYPE
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from tqdm.auto import tqdm

RANDOM_STATE = 1219
BATCH_SIZE = 1024
# One log line per this many optimizer steps.
LOG_EVERY = 500
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(name: str) -> logging.Logger:
    """
    File logger for one training run, written to logs/{name}.log.

    Re-running a notebook cell calls this again, so handlers are added only once:
    otherwise every line would be written twice, then three times, and so on.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"chessml.train.{name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_DIR / f"{name}.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger

def save_checkpoint(model: nn.Module, name: str, config: dict, epoch: int) -> Path:
    """
    Overwrite artifacts/{name}.pt with the model's current weights in fp16.

    Called after every improved epoch so a crashed run still leaves a scoreable model
    behind. 

    Args:
        config: whatever the notebook needs to rebuild the architecture before
            loading the weights, e.g. {'hidden': 512, 'dropout': 0.1}.

    Returns:
        Path of the written file.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.pt"

    # Only floating point tensors are halved.
    state_dict = {
        k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
        for k, v in model.state_dict().items()
    }

    torch.save({"state_dict": state_dict, "config": config, "epoch": epoch}, path)
    return path

def evaluate_epoch(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    positions_gpu: torch.Tensor,
    labels_gpu: torch.Tensor,
    indices_gpu: torch.Tensor,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    n = len(indices_gpu)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = indices_gpu[start:start + batch_size]
            x = positions_gpu[idx]
            y = labels_gpu[idx]
            logits = forward_fn(model, x)
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=1) == y).sum().item()

    return total_loss / n, correct / n


def train_model(
    model: nn.Module,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    number_of_epochs: int,
    train_positions_indices: npt.NDArray[np.int64],
    val_positions_indices: npt.NDArray[np.int64],
    positions: npt.NDArray[POSITION_DTYPE],
    labels: npt.NDArray[LABEL_DTYPE],
    name: str,
    config:dict,
    *,
    batch_size: int = BATCH_SIZE,
    patience: int = 2,
    min_delta: float = 1e-4,
    seed: int = RANDOM_STATE
) -> dict:
    """
    Train one model and return the course-format metrics dict.

    The function returns the weights of the BEST epoch, not the last one.

    Random seeding is split in two. This function sets the seed which affects shuffling
    (torch.randperm). The initial weights are created during model building phase,
    so the caller must call torch.manual_seed(RANDOM_STATE) in order to have determinism.
    """
    # Set a seed that torch.randperm will use
    torch.manual_seed(seed) # TODO: Is this fine?
    
    logger = setup_logging(name)
    device = get_device()
    model = model.to(device)

    metrics = {
        "train_loss": [], "train_accuracy": [], "train_steps": [],
        "val_loss": [], "val_accuracy": [], "val_steps": [],
    }
    training_step = 0

    pbar = tqdm(total=number_of_epochs, desc=f"Training {name}")
    pbar.set_postfix({"loss": -1, "accuracy": -1})

    positions_gpu = torch.from_numpy(np.asarray(positions)).to(device)
    labels_gpu = torch.from_numpy(np.asarray(labels).astype(np.int64)).to(device)
    train_indices_gpu = torch.from_numpy(train_positions_indices).to(device)
    val_indices_gpu = torch.from_numpy(val_positions_indices).to(device)

    best_val_acc = -1.0
    best_model_state = None
    bad_epochs = 0
    
    for epoch in range(number_of_epochs):
        model.train()
        train_indices_gpu_permuted = train_indices_gpu[torch.randperm(len(train_indices_gpu), device=device)]
        for start in range(0, len(train_indices_gpu_permuted), batch_size):
            idx = train_indices_gpu_permuted[start:start+batch_size]
            x = positions_gpu[idx]
            y = labels_gpu[idx] # NOTE: Seems that we can't call it inputs because the argument is called that, but we are changing the type.

            # bfloat16 halves the memory traffic on this GPU.
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = forward_fn(model, x)
                loss = nn.functional.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()   
            optimizer.step()
            if training_step % LOG_EVERY == 0:
                accuracy = (logits.argmax(dim=1) == y).float().mean().item()
                metrics["train_loss"].append(loss.item())
                metrics["train_accuracy"].append(accuracy)
                metrics["train_steps"].append(training_step)
                logger.info(f"step {training_step} loss {loss.item():.4f} acc {accuracy:.4f}")
            training_step += 1


            
        # Calculate accuracy on validation set and hopefully detect overfitting
        val_loss, val_accuracy = evaluate_epoch(model, forward_fn, positions_gpu, labels_gpu, val_indices_gpu, batch_size)

        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["val_steps"].append(training_step)
        logger.info(f"epoch {epoch} val_loss {val_loss:.4f} val_acc {val_accuracy:.4f}")
        pbar.set_postfix({"loss": val_loss, "accuracy": val_accuracy})
        pbar.update(1)

        if val_accuracy > best_val_acc + min_delta:
            best_val_acc = val_accuracy

            # Save the model state that is best currently
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_checkpoint(model, name, config, epoch)
            # Reset bad epoch counter
            bad_epochs = 0
        else:
            # If our current val accuracy is worse than the best, we don't break immediately,
            # but rather if it happens more than ``patience`` times.
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    pbar.close()

    # Load the best model (so that model isn't one of the worse ones from "bad epochs")
    if best_model_state is None:
        raise ValueError("best_model_state is None.")
    
    # Updates the model in-place, so we don't need to return the model 
    model.load_state_dict(best_model_state)
    
    return metrics



# TODO: We should probably add evaluate_epoch docstring
