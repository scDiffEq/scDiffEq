
import torch


class PotentialMixIn(object):
    """Mixin class for potential-based operations.

    This class provides methods for potential-based operations,
    including validation step with gradient computation enabled.

    Attributes:
        None

    Methods:
        __init__: Initialize the PotentialMixIn object.
        validation_step: Perform a validation step with gradient computation enabled.

    """
    def __init__(self):
        """Initialize the PotentialMixIn object.

        Args:
            None

        Returns:
            None

        """
        super().__init__()

    @torch.enable_grad()
    @torch.inference_mode(False)
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        """Perform a validation step with gradient computation enabled.

        This method performs a validation step with gradient computation enabled,
        if the model is not in pre-training mode.

        Args:
            batch: Input batch of data.
            batch_idx (int): Index of the batch.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        """
        if not self.PRETRAIN:
            return self.step(batch, batch_idx, stage="validation")
