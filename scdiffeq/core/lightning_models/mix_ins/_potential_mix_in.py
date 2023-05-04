
import torch


class PotentialMixIn(object):
    def __init__(self):
        super().__init__()

    @torch.enable_grad()
    @torch.inference_mode(False)
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        if not self.PRETRAIN:
            return self.step(batch, batch_idx, stage="validation")