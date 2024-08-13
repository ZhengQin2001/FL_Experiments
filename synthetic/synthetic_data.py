import numpy as np
from scipy.stats import norm

from scipy.stats import norm
def get_probabilities_extended(x_array, param_pa, param_pyxa, param_pxa):
    p_xa = np.zeros([x_array.shape[0], param_pa.shape[0]])
    p_yxa = np.zeros([x_array.shape[0], param_pa.shape[0]])

    for i in np.arange(param_pa.shape[0]):
        z_i = (x_array - param_pxa[i, 0]) / param_pxa[i, 1]
        p_xa[:, i] = norm.pdf(z_i)

        ## build p_yxa
        params = param_pyxa[i]
        th = params[0]
        values = params[1]
        aux = np.zeros([x_array.shape[0]])
        for j in np.arange(th.shape[0]):
            aux[x_array >= th[j]] += 1

        for j in np.arange(values.shape[0]):
            # print(values[j])
            p_yxa[aux == j, i] = values[j]
    return p_xa, p_yxa

def get_bs_optimal(x_array, p_xa, p_yxa, mua, type = 'MSE'):
    ##########################
    # THIS IS FOR BINARY Y!!!
    #########################

    dx = np.max(x_array) - np.min(x_array)
    dx /= x_array.shape[0]

    h = np.sum(p_yxa * p_xa * mua[np.newaxis, :], axis=1) / np.sum(p_xa * mua[np.newaxis, :], axis=1)

    if type == 'MSE':
        risk = p_yxa * 2 * ((1 - h[:, np.newaxis]) ** 2) + (1 - p_yxa) * 2 * ((h[:, np.newaxis]) ** 2)
        risk *= p_xa
        risk = np.sum(risk, axis=0) * dx
    else:
        risk = -p_yxa*np.log(h[:, np.newaxis]) - (1-p_yxa)*np.log((1-h[:, np.newaxis]))
        risk *=  p_xa
        risk = np.sum(risk, axis=0) * dx


    aux = (p_yxa - h[:, np.newaxis]) * p_xa / np.sum(p_xa * mua)
    aux2 = (p_yxa - h[:, np.newaxis])
    drisk = (-4 * aux[:, np.newaxis, :] * aux2[:, :, np.newaxis]) * p_xa[:, :,
                                                                    np.newaxis]  # -4 cause is binary so its X2
    drisk = np.sum(drisk, axis=0) * dx

    return h, risk, drisk

