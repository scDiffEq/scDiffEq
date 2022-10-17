


from abc import ABC, abstractmethod
from ._integrators import credential_handoff


class BaseBatchForward(ABC):
    def __init__(self, func, loss_function):
        """To-do: add docs."""
        self.integrator, self.func_type = credential_handoff(func)
        self.loss_function = loss_function
        self.func = func

    @abstractmethod
    def __parse__(self):
        pass

    @abstractmethod
    def __inference__(self):
        pass

    @abstractmethod
    def __loss__(self):
        pass
    
    @abstractmethod
    def __call__(self, model, batch, stage, **kwargs):
        pass


class BatchForward(BaseBatchForward):
    
    def _sum_norm(self, W):
        return W / W.sum(1)[:, None]

    def _format_sinkhorn_weights(self, W, W_hat):
        self.W, self.W_hat = self._sum_norm(W), self._sum_norm(W_hat)

    def __parse__(self, batch):
        self.t = batch[0].unique()
        if len(batch) >= 3:
            W = batch[2].transpose(1, 0)
        
        if len(batch) == 4:
            W_hat = batch[3].transpose(1, 0)
            self._format_sinkhorn_weights(W, W_hat)
        
        self.X = batch[1].transpose(1, 0)
        self.X0 = self.X[0]

    def __inference__(self, dt, **kwargs):
        self.X_hat = self.integrator(self.func, self.X0, ts=self.t, dt=dt, **kwargs)
        return self.X_hat

    def __loss__(self):
        return self.loss_function(
            self.X_hat.contiguous(), self.X.contiguous(), self.W, self.W_hat,
        )

    def __log__(self, model, stage, loss):
        for n, i in enumerate(range(len(self.t))[-len(loss):]):
            model.log("{}_{}_loss".format(stage, self.t[i]), loss[n])

    def __call__(self, model, batch, stage, **kwargs):
        """
        By default, __call___ will run:
        (1) __parse__()
        (2) __inference__()
        (3) __loss__()
        (4) __log__()
        Finally, it returns the output of loss.
        """
        self.__parse__(batch)
        X_hat = self.__inference__(dt=kwargs["dt"])
        if stage == "predict":
            return X_hat
        loss  = self.__loss__()
        self.__log__(model, stage, loss)
        return loss.sum()