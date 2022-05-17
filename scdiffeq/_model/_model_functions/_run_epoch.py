
from ._forward_integrate import _forward_integrate_by_time_group
from ._loss import _loss

def _run_epoch(
    FormattedData, func, optimizer, status_file, test, loss_function, dry, silent, epoch, device,
):

    X_pred = _forward_integrate_by_time_group(FormattedData, optimizer, func, test, device)
    loss_df = _loss(
        X_pred,
        optimizer,
        loss_function,
        test,
        epoch,
        status_file,
        dry,
        silent,
    )
    
    return X_pred, loss_df