import torch

from ..._tools._forward_integrate import _forward_integrate

def _forward_integrate_over_test_data(X_pred_, FormattedData, key, func, device):

    with torch.no_grad():
        (
            X_pred_[key]["X_pred_test"],
            X_pred_[key]["X_test"],
        ) = _forward_integrate(
            func,
            X=FormattedData[key]["X_test"],
            t=FormattedData[key]["t"],
            device=device,
        )
        X_pred_[key]["X_test_idx"] = FormattedData[key]["X_test_idx"]

        return X_pred_


def _forward_integrate_by_time_group(FormattedData, optimizer, func, test, device):

    """"""

    X_pred_ = {}

    for key in FormattedData.keys():
        X_pred_[key] = {}
        X_pred, X = _forward_integrate(
            func,
            X=FormattedData[key]["X_train"],
            t=FormattedData[key]["t"],
            device=device,
        )
        X_pred_[key]["X_pred"], X_pred_[key]["X"] = X_pred, X
        X_pred_[key]["X_train_idx"] = FormattedData[key]["X_train_idx"]
        if test:
            if not FormattedData[key]["X_test"] is None:
                _forward_integrate_over_test_data(
                    X_pred_, FormattedData, key, func, device
                )
            else:
                continue
    return X_pred_