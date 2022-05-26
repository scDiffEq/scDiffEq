

class _Loss:
    def __init__(self, reconst_loss_func, reparam_loss_func, device):

        self._ReconstLoss = reconst_loss_func
        self._ReparamLoss = reparam_loss_func

        self._reconst_loss = 0
        self._reparam_loss = 0

    def Reconstruction(self, X_pred, X):
        self._reconst_loss = self._ReconstLoss(X_pred, X)

    def Reparameterization(self, mu, log_var):
        self._reparam_loss = self._ReparamLoss(mu, log_var)

    def total(self):
        self._total_loss = self._reconst_loss + self._reparam_loss


def _calculate_loss(X_pred, mu, log_var, reconst_loss_func, reparam_loss_func, device):

    loss = _Loss(reconst_loss_func, reparam_loss_func, device)
    loss.Reconstruction(X_pred, X)

    if (mu != None) and (log_var != None):
        loss.Reparameterization(mu, log_var)

    loss.total()

    return loss