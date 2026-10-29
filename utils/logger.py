class Logger:
    """Thin wrapper: pass backend='wandb', 'tensorboard', or None (no-op)."""

    def __init__(self, backend: str = None, run_name: str = "run", log_dir: str = "./runs"):
        self.backend = backend
        self._impl = None
        if backend == "wandb":
            import wandb
            self._impl = wandb.init(project="dataprune-bench", name=run_name, reinit=True)
        elif backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self._impl = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

    def log(self, metrics: dict):
        if self.backend == "wandb":
            self._impl.log(metrics)
        elif self.backend == "tensorboard":
            step = metrics.get("epoch", 0)
            for k, v in metrics.items():
                if k != "epoch":
                    self._impl.add_scalar(k, v, step)
        # backend is None -> no-op

    def close(self):
        if self.backend == "tensorboard" and self._impl is not None:
            self._impl.close()
        if self.backend == "wandb" and self._impl is not None:
            self._impl.finish()
