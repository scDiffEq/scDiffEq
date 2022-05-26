

def _pass_to_VAE_scDiffEq_model(model, X0, t):

    model.encode(X0)
    model.reparameterize()
    model.forward_int(t)

    return model.decode(), model._mu, model._log_var


def _pass_to_scDiffEq_model(model, X0, t):

    return model.forward_int(X0, t)


def _pass_to_model(model, X0, t, VAE):

    if not VAE == None:
        return _pass_to_VAE_scDiffEq_model(model, X0, t, VAE)
    else:
        return _pass_to_scDiffEq_model(model, X0, t, VAE)


def _model_forward(model, X, t, VAE, reconst_loss_func, reparam_loss_func, device):

    X0 = X[0]

    X_pred, mu, log_var = _pass_to_model(model, X0, t, VAE)
    loss = _calculate_loss(
        X_pred, mu, log_var, reconst_loss_func, reparam_loss_func, device
    )

    return X_pred, loss


def _batched_no_grad_model_pass(
    X,
    model,
    t,
    VAE,
    reconst_loss_func,
    reparam_loss_func,
    device,
):

    batched_data = _get_batches(X)
    batched_loss = []
    with torch.no_grad():
        for X_batch in batched_data:
            X_pred, loss = _model_forward(
                self._model,
                X_batch,
                t,
                self._VAE,
                self._reconst_loss_func,
                self._reparam_loss_func,
                self._device,
            )

            loss = loss.detach().cpu()
            batched_loss.append(loss)

    return batched_loss


def _batched_training_model_pass(
    X, model, optimizer, t, VAE, reconst_loss_func, reparam_loss_func, device
):

    batched_data = _get_batches(X)

    for X_batch in batched_data:
        X_pred, loss = _model_forward(
            model,
            X_batch,
            t,
            VAE,
            reconst_loss_func,
            reparam_loss_func,
            device,
        )
        loss.sum().backward()
        optimizer.step()
        loss = loss.detach().cpu()

        return X_pred, loss.item()