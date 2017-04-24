import numpy as np

from preprocessing import LabelBinarizer
from sklearn import svm


def l1_min_c(X, y, loss='squared_hinge', fit_intercept=True,
             intercept_scaling=1.0):
   
    X = check_array(X, accept_sparse='csc')
    check_consistent_length(X, y)

    Y = LabelBinarizer(neg_label=-1).fit_transform(y).T
    # maximum absolute value over classes and features
    den = np.max(np.abs(safe_sparse_dot(Y, X)))
    if fit_intercept:
        bias = intercept_scaling * np.ones((np.size(y), 1))
        den = max(den, abs(np.dot(Y, bias)).max())

    if den == 0.0:
        raise ValueError('Ill-posed l1_min_c calculation: l1 will always '
                         'select zero coefficients for this data')
    if loss == 'squared_hinge':
        return 0.5 / den
    else:  # loss == 'log':
        return 2.0 / den

