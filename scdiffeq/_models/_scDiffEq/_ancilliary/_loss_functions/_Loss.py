

class _Loss:
    def __init__(self, reconst_loss_func, reparam_loss_func, device):

        self._ReconstLoss = reconst_loss_func
        self._ReparamLoss = reparam_loss_func

        self._reconst_loss = 0
        self._reparam_loss = 0

    def Reconstruction(self, X_pred, X):
#         print("loss func input shapes:", X_pred.shape, X.shape)
        self._reconst_loss = self._ReconstLoss(X_pred[1:], X[1:]) # exclude the first time point
    #.abs() # .reshape(-1, 50)
#         print(self._reconst_loss)
#         print(" - Recon loss is: {}".format(self._reconst_loss))

    def Reparameterization(self, mu, log_var):
        self._reparam_loss = self._ReparamLoss(mu, log_var).abs()

    def total(self, scaling_factor = 1):
        
        
        # seems we may need to do some scaling??
    
#         scaling_factor = self._reconst_loss / self._reparam_loss
        
        self._reconst_loss = (self._reconst_loss / scaling_factor)
#         print("RECON: {} |  REPARAM: {}".format(self._reconst_loss, self._reparam_loss))

#         print(" - Reparam loss: {}".format(self._reparam_loss))
        self.total_loss = self._reconst_loss + self._reparam_loss
        self.total_loss.requres_grad = True
        

def _calculate_loss(X_pred, X, mu, log_var, reconst_loss_func, reparam_loss_func, device):
    
#     print(X_pred.shape, X.shape)
    
    loss = _Loss(reconst_loss_func, reparam_loss_func, device)
    loss.Reconstruction(X_pred, X)
    
#     print("mu: {} | log_var: {}".format(mu, log_var))

    if (mu != None) and (log_var != None):
        loss.Reparameterization(mu, log_var)

    loss.total()

    return loss