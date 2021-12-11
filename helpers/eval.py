import numpy as np
from .metrics import get_auc, get_aucpr
from sklearn.metrics import confusion_matrix, classification_report


def mad_score(points):
    """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)

    return 0.6745 * ad / mad


def use_mad_threshold(model, X, threshold: float = 3.5):
    y_prob = -model.score_samples(X)
    z_scores = mad_score(y_prob)
    return z_scores > threshold


def eval_model(X, y, model, is_dl: bool = False, use_mad: bool = False):
    if is_dl:
        y_pred = y_prob = model.get_outliers(X)
    else:  # IF
        if use_mad:
            y_pred = use_mad_threshold(model, X, threshold=1.65)
        else:
            y_pred = model.predict(X)
            y_pred = np.where(y_pred == 1, 0, 1)
        y_prob = -model.score_samples(X)

    # get metrics
    _, _, auc_score = get_auc(y, y_prob)
    _, _, aucpr_score = get_aucpr(y, y_prob)

    res_scores = {'auc': np.round(auc_score, 3), 'aucpr': np.round(aucpr_score, 3)}

    (res_scores['tn'], res_scores['fp'], res_scores['fn'],
     res_scores['tp']) = confusion_matrix(y, y_pred).ravel()

    target_names = ['Normal', 'Anomalies']
    report_dict = classification_report(y, y_pred, target_names=target_names,
                                        output_dict=True, digits=2)
    for label, values in report_dict.items():
        if type(values) == dict:
            for key, value in values.items():
                res_scores[f'{label}_{key}'] = np.round(value, 3)
        else:
            res_scores[label] = np.round(values, 3)

    return res_scores
