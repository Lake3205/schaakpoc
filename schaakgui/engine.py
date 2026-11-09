"""Qt-independent training engine (run inside a QThread by :mod:`schaakgui.worker`)."""
from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import Lineage, build_model, load_checkpoint, model_from_checkpoint, save_checkpoint
from .data import (
    MultiFolderImageDataset,
    build_eval_transform,
    build_train_transform,
    discover_classes,
    enabled_augmentations,
)
from .metrics import confidence_curves, confusion_matrix, macro_precision_recall
from .projects import AugmentationConfig, QuotaConfig, TrainingConfig
from .quota import EarlyStopping, QuotaTracker

VAL_SPLIT = 0.2


class TrainingStopped(Exception):
    """Raised inside an epoch when the user requested a stop."""


@dataclass
class TrainJob:
    train_folders: list[str]
    test_folders: list[str]
    output_dir: str
    shuffle: bool = True
    seed: Optional[int] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    resume_checkpoint: Optional[str] = None


@dataclass
class Callbacks:
    log: Callable[[str], None] = lambda msg: None
    batch_progress: Callable[[int, int], None] = lambda i, n: None
    epoch_progress: Callable[[int, int], None] = lambda e, n: None
    evaluated: Callable[[dict], None] = lambda payload: None
    milestone: Callable[[dict], None] = lambda m: None
    paused: Callable[[str], None] = lambda checkpoint_path: None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, num_classes: int,
             criterion: Optional[nn.Module] = None) -> dict:
    """Validation loss, accuracy, confidences and confusion matrix."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss, y_true, y_pred, conf = 0.0, [], [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += criterion(logits, targets).item() * len(targets)
        probs = torch.softmax(logits, dim=1)
        top_conf, top_idx = probs.max(dim=1)
        y_true += targets.cpu().tolist()
        y_pred += top_idx.cpu().tolist()
        conf += top_conf.cpu().tolist()
    n = max(1, len(y_true))
    correct = np.asarray(y_true) == np.asarray(y_pred)
    return {
        "loss": total_loss / n,
        "accuracy": float(correct.mean()) if len(y_true) else 0.0,
        "mean_confidence": float(np.mean(conf)) if conf else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": conf,
        "confusion": confusion_matrix(y_true, y_pred, num_classes),
    }


class Trainer:
    def __init__(self, job: TrainJob, callbacks: Optional[Callbacks] = None):
        self.job = job
        self.cb = callbacks or Callbacks()
        self._stop = threading.Event()
        self._running = threading.Event()
        self._running.set()

    # ---- control (thread safe) -------------------------------------------------
    def request_stop(self):
        self._stop.set()
        self._running.set()

    def pause(self):
        self._running.clear()

    def resume(self):
        self._running.set()

    @property
    def is_paused(self) -> bool:
        return not self._running.is_set()

    def _checkpoint_pause(self, save_review: Callable[[], str]):
        if not self._running.is_set():
            path = save_review()
            self.cb.log(f"Training paused. Current weights saved to {path}")
            self.cb.paused(path)
            while not self._running.wait(0.1):
                pass
            if not self._stop.is_set():
                self.cb.log("Training resumed.")
        if self._stop.is_set():
            raise TrainingStopped()

    # ---- main loop -------------------------------------------------------------
    def run(self) -> dict:
        job, cfg, log = self.job, self.job.training, self.cb.log
        out = Path(job.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
        if job.seed is not None:
            random.seed(job.seed)
            np.random.seed(job.seed)
            torch.manual_seed(job.seed)
        if not job.train_folders:
            raise ValueError("Select at least one training folder.")

        ckpt = load_checkpoint(job.resume_checkpoint) if job.resume_checkpoint else None
        if ckpt is not None:
            classes = list(ckpt["classes"])
            architecture, image_size = ckpt["architecture"], int(ckpt["image_size"])
            lineage = Lineage.from_dict(ckpt.get("lineage"))
            unknown = sorted(set(discover_classes(job.train_folders)) - set(classes))
            if unknown:
                raise ValueError(f"Training folders contain classes not known by the checkpoint: "
                                 f"{', '.join(unknown)}")
            log(f"Resuming from {job.resume_checkpoint} ({lineage.total_epochs_completed} epochs "
                f"already trained, architecture {architecture}, image size {image_size}).")
        else:
            classes = discover_classes(job.train_folders)
            architecture, image_size = cfg.architecture, cfg.image_size
            lineage = Lineage()
        if not classes:
            raise ValueError("No class sub-folders with images found in the training folders.")
        log(f"Classes ({len(classes)}): {', '.join(classes)}")

        train_tf = build_train_transform(job.augmentation, image_size)
        eval_tf = build_eval_transform(image_size)
        log("Augmentations: " + (", ".join(enabled_augmentations(job.augmentation)) or "none"))
        if job.test_folders:
            train_ds = MultiFolderImageDataset(job.train_folders, classes, train_tf)
            val_ds = MultiFolderImageDataset(job.test_folders, classes, eval_tf)
            if val_ds.skipped_classes:
                log("Warning: skipping test classes unknown to the model: "
                    + ", ".join(sorted(val_ds.skipped_classes)))
        else:
            full = MultiFolderImageDataset(job.train_folders, classes)
            train_ds, val_ds = full.split(VAL_SPLIT, job.seed, train_tf, eval_tf)
            log(f"No test folders selected: holding out {VAL_SPLIT:.0%} of the training pool for validation.")
        if len(train_ds) < 2 or len(val_ds) == 0:
            raise ValueError("Training (at least 2 images) and validation sets must contain images.")
        log(f"Training samples: {len(train_ds)}, validation samples: {len(val_ds)}, "
            f"shuffle: {'on' if job.shuffle else 'off'}, seed: {job.seed}")

        generator = None
        if job.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(job.seed)
        loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                           pin_memory=device.type == "cuda")
        # A trailing batch of a single sample breaks BatchNorm layers in training mode.
        drop_last = len(train_ds) > cfg.batch_size and len(train_ds) % cfg.batch_size == 1
        train_loader = DataLoader(train_ds, shuffle=job.shuffle, generator=generator,
                                  drop_last=drop_last, **loader_args)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

        if ckpt is not None:
            model = model_from_checkpoint(ckpt)
        else:
            try:
                model = build_model(architecture, len(classes), cfg.pretrained)
            except Exception as e:  # e.g. no internet to download pretrained weights
                if not cfg.pretrained:
                    raise
                log(f"Could not load pretrained weights ({e}); training from scratch.")
                model = build_model(architecture, len(classes), False)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        if ckpt is not None and ckpt.get("optimizer_state"):
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except (ValueError, KeyError) as e:
                log(f"Optimizer state not restored ({e}).")
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate
        criterion = nn.CrossEntropyLoss()

        quota = QuotaTracker(job.quota, lineage.quota_history)
        stopper = None
        if job.quota.early_stopping:
            mode = "max" if job.quota.monitor == "val_acc" else "min"
            stopper = EarlyStopping(job.quota.patience, mode)

        start_epoch = lineage.total_epochs_completed
        session = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "start_epoch": start_epoch,
            "end_epoch": start_epoch,
            "train_folders": list(job.train_folders),
            "test_folders": list(job.test_folders),
            "augmentations": enabled_augmentations(job.augmentation),
            "resumed_from": job.resume_checkpoint,
            "output_dir": str(out.resolve()),
            "stop_reason": None,
        }
        lineage.sessions.append(session)
        best_acc = lineage.best["val_acc"] if lineage.best else -1.0
        last_path = out / "last.pt"

        def save(path, epoch, metrics=None):
            return save_checkpoint(path, model, classes, architecture, image_size, lineage,
                                   optimizer, epoch, metrics)

        def save_review():
            return str(save(out / "paused.pt", lineage.total_epochs_completed).resolve())

        stop_reason, last_metrics = "completed", {}
        for i in range(cfg.epochs):
            epoch = start_epoch + i + 1
            try:
                self._checkpoint_pause(save_review)
                train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion,
                                                          device, save_review)
            except TrainingStopped:
                stop_reason = "user"
                log("Training stopped by user.")
                break
            ev = evaluate(model, val_loader, device, len(classes), criterion)
            precision, recall = macro_precision_recall(ev["confusion"])
            lr = optimizer.param_groups[0]["lr"]
            lineage.total_epochs_completed = epoch
            session["end_epoch"] = epoch
            last_metrics = {
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": ev["loss"], "val_acc": ev["accuracy"],
                "mean_confidence": ev["mean_confidence"], "precision": precision,
                "recall": recall, "lr": lr,
            }
            lineage.history.append(last_metrics)
            milestones = quota.update(epoch, ev["accuracy"], ev["mean_confidence"])
            lineage.quota_history = quota.history
            log(f"Epoch {epoch} ({i + 1}/{cfg.epochs}): train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} val_loss={ev['loss']:.4f} val_acc={ev['accuracy']:.3f} "
                f"conf={ev['mean_confidence']:.3f} P={precision:.3f} R={recall:.3f} lr={lr:.2e}")
            log(f"Quota status: {quota.describe()}")
            for m in milestones:
                log(f"Milestone: {m['message']}")
                self.cb.milestone(m)

            if ev["accuracy"] > best_acc:
                best_acc = ev["accuracy"]
                best_path = (out / "best.pt").resolve()
                lineage.best = {"epoch": epoch, "path": str(best_path), "val_acc": ev["accuracy"],
                                "val_loss": ev["loss"]}
                save(best_path, epoch, last_metrics)
                log(f"New best model (val_acc={best_acc:.3f}) saved to {best_path}")
            every = int(job.quota.snapshot_every)
            if every > 0 and epoch % every == 0:
                snap_path = (out / "snapshots" / f"epoch_{epoch:04d}.pt").resolve()
                lineage.snapshots.append({"epoch": epoch, "path": str(snap_path),
                                          "val_acc": ev["accuracy"]})
                save(snap_path, epoch, last_metrics)
                log(f"Snapshot saved to {snap_path}")
            save(last_path, epoch, last_metrics)

            self.cb.evaluated({
                **last_metrics,
                "classes": classes,
                "confusion": ev["confusion"].tolist(),
                "history": list(lineage.history),
                "curves": confidence_curves(ev["confidences"],
                                            np.asarray(ev["y_true"]) == np.asarray(ev["y_pred"])),
                "quota_status": dict(quota.current),
                "checkpoint": str(last_path.resolve()),
            })
            self.cb.epoch_progress(i + 1, cfg.epochs)

            if job.quota.stop_on_quota and quota.all_met():
                stop_reason = "quota"
                log("All quota targets satisfied - stopping training.")
                break
            if stopper is not None:
                value = ev["accuracy"] if stopper.mode == "max" else ev["loss"]
                if stopper.step(value):
                    stop_reason = "early_stopping"
                    log(f"Early stopping: {job.quota.monitor} did not improve for "
                        f"{stopper.patience} epochs.")
                    break

        session["stop_reason"] = stop_reason
        session["ended_at"] = datetime.now().isoformat(timespec="seconds")
        trained = session["end_epoch"] > start_epoch
        if trained:
            save(last_path, lineage.total_epochs_completed, last_metrics)
        else:
            lineage.sessions.pop()
        paused = out / "paused.pt"
        paused.unlink(missing_ok=True)
        log(f"Finished ({stop_reason}). Total epochs trained: {lineage.total_epochs_completed}")
        return {
            "stop_reason": stop_reason,
            "total_epochs_completed": lineage.total_epochs_completed,
            "epochs_this_session": session["end_epoch"] - start_epoch,
            "last_checkpoint": str(last_path.resolve()) if trained else None,
            "best_checkpoint": lineage.best["path"] if lineage.best else None,
            "output_dir": str(out.resolve()),
            "metrics": last_metrics,
            "milestones": [m["message"] for m in lineage.quota_history],
            "finished_at": session["ended_at"],
        }

    def _train_epoch(self, model, loader, optimizer, criterion, device, save_review):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        n_batches = len(loader)
        for b, (images, targets) in enumerate(loader):
            self._checkpoint_pause(save_review)
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += len(targets)
            self.cb.batch_progress(b + 1, n_batches)
        return total_loss / max(1, seen), correct / max(1, seen)
