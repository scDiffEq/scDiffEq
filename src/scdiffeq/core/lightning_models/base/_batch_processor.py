
# -- import packages: ---------------------------------------------------------
import ABCParse
import torch
import numpy as np


# -- set typing: --------------------------------------------------------------
from typing import Union, List


class BatchRepresentation(object):
    """Representation of a batch of data.

    This class provides a textual representation of a batch of data, including its header,
    data tensor shapes, and fate index shape if available.

    Attributes:
        header (str): Header for the batch representation.
        X (str): Representation of the data tensor shape.
        W (str): Representation of the weight tensor shape.
        W_hat (str): Representation of the predicted weight tensor shape.
        F_idx (str): Representation of the fate index tensor shape if available, empty string otherwise.

    Methods:
        __call__: Generate the textual representation of the batch.

    """
    
    def __init__(self, batch):
        
        """Initialize the BatchRepresentation object.

        Args:
            batch: Batch of data.

        Returns:
            None

        """

        self.header = "🍪 LitDataBatch"
        self.X = f"- X: {list(batch.X.shape)}"
        self.W = f"- W: {list(batch.W.shape)}"
        self.W_hat = f"- W_hat: {list(batch.W_hat.shape)}"
        if not batch.F_idx is None:
            self.F_idx = f"- F_idx: {list(batch.F_idx.shape)}"
        else:
            self.F_idx = ""

    def __call__(self):
        """Generate the textual representation of the batch.

        Returns:
            str: Textual representation of the batch.

        """
        return "\n".join([self.header, self.X, self.W, self.W_hat, self.F_idx])


class BatchProcessor(ABCParse.ABCParse):    
    """Batch processor class for data preprocessing.

    This class provides methods for preprocessing a batch of data including sum normalization,
    and properties for accessing batch-related information.
    
    Batch Items:
    ------------
    0. t - time. always required
    1. X - data. tensor of size: [batch_size, len(t), n_dim]. always required.
    2. W - weight. tensor of size: [batch_size, len(t), 1]
    3. F_idx - fate_idx. tensor of size: [batch_Size, len(t), 1]

    Attributes:
        None

    Methods:
        _sum_normalize: Normalize the input tensor by sum along a specified axis.
        __repr__: Generate the textual representation of the batch.

    Properties:
        device: Get the device of the batch.
        n_batch_items: Get the number of items in the batch.
        batch_size: Get the batch size.
        t: Get unique time steps in the batch.
        X: Get the data tensor.
        X0: Get the initial data tensor.
        W_hat: Get the predicted weight tensor.
        W: Get the normalized weight tensor.
        F_idx: Get the fate index tensor if available.

    """

    def __init__(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Initialize the BatchProcessor object.

        Args:
            batch (List[torch.Tensor]): List of tensors in the batch.
            batch_idx (int): Index of the batch.

        Returns:
            None

        """

        self.__parse__(locals(), private=["batch"])

    def _sum_normalize(
        self, X: Union[torch.Tensor, np.ndarray], sample_axis=1
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Normalize the input tensor by sum along a specified axis.

        Args:
            X (Union[torch.Tensor, np.ndarray]): Input tensor.
            sample_axis (int): Axis along which to perform normalization. Default is 1.

        Returns:
            Union[torch.Tensor, np.ndarray]: Normalized tensor.

        """
        
        return X / X.sum(sample_axis)[:, None]

    @property
    def device(self):
        """Get the device of the batch."""
        return self._batch[0].device

    @property
    def n_batch_items(self) -> int:
        """Get the number of items in the batch."""
        return len(self._batch)

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self._batch[0].shape[0]

    @property
    def t(self) -> torch.Tensor:
        """Get unique time steps in the batch."""
        return self._batch[0].unique()

    @property
    def X(self) -> torch.Tensor:
        """Get the cell state tensor."""
        return self._batch[1].transpose(1, 0).contiguous()

    @property
    def X0(self) -> torch.Tensor:
        """Get the initial cell state tensor."""
        return self.X[0]

    @property
    def W_hat(self) -> torch.Tensor:
        """Predicted weight tensor.

        If the predicted weight tensor is not cached, it will be computed and cached.
        """
        if not hasattr(self, "_W_hat"):
            if self.n_batch_items >= 3:
                W = self._batch[2].transpose(1, 0)
                for i in range(1, len(self.t)):
                    W[i] = torch.exp((self.t[i] - self.t[0]) * W[0])
                W[0] = torch.ones_like(W[0])
                self._W_hat = self._sum_normalize(W, sample_axis=1).contiguous()
            else:
                self._W_hat = self._sum_normalize(
                    torch.ones([len(self.t), self.batch_size, 1], device=self.device)
                ).contiguous()
        return self._W_hat

    @property
    def W(self) -> torch.Tensor:
        """Normalized weight tensor."""
        return self._sum_normalize(torch.ones_like(self.W_hat)).contiguous()

    @property
    def F_idx(self):
        """Fate index tensor if available."""
        if self.n_batch_items >= 4:
            return (
                self._batch[3]
                .transpose(1, 0)[0]
                .detach()
                .cpu()
                .numpy()
                .flatten()
                .astype(int)
                .astype(str)
            )

    def __repr__(self) -> str:
        """Generate the textual representation of the batch."""
        return BatchRepresentation(self)()
