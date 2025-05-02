# -- import packages: ----------------------------------------------------------
import gc
import lightning
import os
import torch


# -- operational cls: ----------------------------------------------------------
class MemoryMonitor(lightning.pytorch.callbacks.Callback):

    def __init__(self, log_gpu=True):

        import psutil

        super().__init__()

        self.log_gpu = log_gpu
        self.process = psutil.Process(os.getpid())

    def _log_cpu_ram(self, trainer):

        import wandb

        mem = self.process.memory_info().rss / 1024**2  # in MB
        print(f"[MemoryLogger] Epoch {trainer.current_epoch}: CPU RAM: {mem:.2f} MB")
        wandb.log({"Memory/CPU_RAM_MB": mem})

    def _log_gpu_ram(self, trainer, lit_module):
        if self.log_gpu and lit_module.device.type == "cuda":
            gpu_mem_allocated = lit_module.device
            mem_allocated = torch.cuda.memory_allocated(lit_module.device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(lit_module.device) / 1024**2
            print(
                f"[MemoryLogger] Epoch {trainer.current_epoch}: GPU Allocated: {mem_allocated:.2f} MB | Reserved: {mem_reserved:.2f} MB"
            )
            wandb.log(
                {
                    "Memory/GPU_Allocated_MB": mem_allocated,
                    "Memory/GPU_Reserved_MB": mem_reserved,
                }
            )

    def _report_object_memories(self, trainer):
        objs = gc.get_objects()
        print(
            f"[MemoryLogger] Epoch {trainer.current_epoch}: Number of tracked objects: {len(objs)}"
        )

    def on_train_epoch_end(self, trainer, lit_module):

        self._log_cpu_ram(trainer=trainer)
        self._log_gpu_ram(trainer=trainer, lit_module=lit_module)
        self._report_object_memories(trainer=trainer)
