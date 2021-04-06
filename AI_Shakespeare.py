import numpy as np
import random

# Data Preprocessing
data = open('Shakespeare_short.txt', 'r').read()
data = data.lower()
chars = list(set(data))
chars.extend(['<B>', '<E>'])
data_size, vocab_size = len(data), len(chars)
print(data_size)
chars = sorted(chars)
print(vocab_size)
char_to_ind = { ch:i for i,ch in enumerate(chars) }
ind_to_char = { i:ch for i,ch in enumerate(chars) }

# Auxillary functions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    t = (x > 0).astype(int)
    return x * t

def clip(gradients, maxValue): # for the exploding gradients problem
    dWf, dWi, dWc, dWo, dWy = gradients['dWf'], gradients['dWi'], gradients['dWc'], gradients['dWo'], gradients['dWy']
    dbf, dbi, dbc, dbo, dby = gradients['dbf'], gradients['dbi'], gradients['dbc'], gradients['dbo'], gradients['dby']

    for gradient in [dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby]:
        np.clip(gradient, a_min = -maxValue, a_max = maxValue, out = gradient)

    gradients = {"dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo, "dWy": dWy,"dby": dby}

    return gradients

def print_sample(sample_ind, ind_to_char):
    txt = ''.join(ind_to_char[ind] for ind in sample_ind)
    print(txt)

# Initialize Parameters, Gradients, Adam Variables
def initialize_parameters(n_x, n_a, n_y): # Xavier Initialization

    parameters = dict()

    parameters['Wf'] = np.random.randn(n_a, n_a + n_x) * np.sqrt(6 / (2 * n_a + n_x))
    parameters['Wi'] = np.random.randn(n_a, n_a + n_x) * np.sqrt(6 / (2 * n_a + n_x))
    parameters['Wc'] = np.random.randn(n_a, n_a + n_x) * np.sqrt(6 / (2 * n_a + n_x))
    parameters['Wo'] = np.random.randn(n_a, n_a + n_x) * np.sqrt(6 / (2 * n_a + n_x))
    parameters['Wy'] = np.random.randn(n_y, 2 * n_a) * np.sqrt(6 / (n_y + 2 * n_a))
    parameters['bf'] = np.zeros((n_a, 1))
    parameters['bi'] = np.zeros((n_a, 1))
    parameters['bc'] = np.zeros((n_a, 1))
    parameters['bo'] = np.zeros((n_a, 1))
    parameters['by'] = np.zeros((n_y, 1))

    return parameters

def create_gradients(n_x, n_a, n_y, m, T_x):

    gradients = dict()

    gradients['dx'] = np.zeros((n_x, m, T_x))
    gradients['da0'] = np.zeros((n_a, m))
    gradients['dWf'] = np.zeros((n_a, n_a + n_x))
    gradients['dWi'] = np.zeros((n_a, n_a + n_x))
    gradients['dWc'] = np.zeros((n_a, n_a + n_x))
    gradients['dWo'] = np.zeros((n_a, n_a + n_x))
    gradients['dWy'] = np.zeros((n_y, 2 * n_a))
    gradients['dbf'] = np.zeros((n_a, 1))
    gradients['dbi'] = np.zeros((n_a, 1))
    gradients['dbc'] = np.zeros((n_a, 1))
    gradients['dbo'] = np.zeros((n_a, 1))
    gradients['dby'] = np.zeros((n_y, 1))

    return gradients

def initialize_v_and_s(parameters): # Adam Variables

    v = dict()
    s = dict()

    v["vdWf"] = np.zeros(parameters["Wf"].shape)
    v["vdbf"] = np.zeros(parameters["bf"].shape)
    v["vdWi"] = np.zeros(parameters["Wi"].shape)
    v["vdbi"] = np.zeros(parameters["bi"].shape)
    v["vdWc"] = np.zeros(parameters["Wc"].shape)
    v["vdbc"] = np.zeros(parameters["bc"].shape)
    v["vdWo"] = np.zeros(parameters["Wo"].shape)
    v["vdbo"] = np.zeros(parameters["bo"].shape)
    v["vdWy"] = np.zeros(parameters["Wy"].shape)
    v["vdby"] = np.zeros(parameters["by"].shape)

    s["sdWf"] = np.zeros(parameters["Wf"].shape)
    s["sdbf"] = np.zeros(parameters["bf"].shape)
    s["sdWi"] = np.zeros(parameters["Wi"].shape)
    s["sdbi"] = np.zeros(parameters["bi"].shape)
    s["sdWc"] = np.zeros(parameters["Wc"].shape)
    s["sdbc"] = np.zeros(parameters["bc"].shape)
    s["sdWo"] = np.zeros(parameters["Wo"].shape)
    s["sdbo"] = np.zeros(parameters["bo"].shape)
    s["sdWy"] = np.zeros(parameters["Wy"].shape)
    s["sdby"] = np.zeros(parameters["by"].shape)

    return v, s

# LSTM blocks
def lstm_cell_forward(xt, a_prev, c_prev, parameters):

    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    concat = np.concatenate((a_prev, xt), axis = 0)

    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = (ft * c_prev) + (it * cct)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, cache

def lstm_forward(X, a0, parameters, vocab_size, layer = 'later'):
    caches = []

    Wy = parameters['Wy']
    if layer == 'first':
        m, T_x = X.shape
        n_x = vocab_size
    if layer == 'later':
        n_x, m, T_x = X.shape

    n_y, n_a = Wy.shape
    n_a //= 2

    if layer == 'first':
        x = np.zeros((n_x, m, T_x))
    else:
        x = X
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))

    a_next = a0
    c_next = np.zeros((n_a, m))

    for t in range(T_x):
        xt = x[:, :, t]
        if layer == 'first':
            for j in range(m):
                xt[int(X[j, t]), j] = 1
        a_next, c_next, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:,:,t] = a_next
        c[:,:,t]  = c_next
        caches.append(cache)

    caches = (caches, x)

    return a, c, caches

def BiLSTM_forward_first(X, parameters, vocab_size, activation = 'relu'):
    m, T_x = X.shape
    n_a = parameters['bf'].shape[0]
    x = np.zeros((m, T_x + 1))
    x_ = np.zeros((m, T_x + 1))
    Wy = parameters["Wy"]
    by = parameters["by"]
    n_y = by.shape[0]
    ya = np.zeros((n_y, m, T_x))

    for i in range(m):
        xi = list(X[i])
        x[i,:] = np.array([char_to_ind['<B>']] + xi)
        x_[i,:] = np.array([char_to_ind['<E>']] + list(reversed(xi)))

    a0 = np.zeros((n_a, m))
    a0_ = np.zeros((n_a, m))
    a, c, caches = lstm_forward(x, a0, parameters, vocab_size, layer = 'first')
    a_, c_, caches_ = lstm_forward(x_, a0_, parameters, vocab_size, layer = 'first')
    a = a[:,:, : T_x]
    a_ = a_[:,:, : T_x]

    ao = np.concatenate((a, a_), axis = 0)
    for t in range(T_x):
        ya[:,:,t] = relu(np.dot(Wy, ao[:,:,t]) + by)

    return ya, ao, caches, caches_

def BiLSTM_forward_later(X, parameters, vocab_size, activation = 'relu'):
    n_x, m, T_x = X.shape
    n_a = parameters['bf'].shape[0]
    Wy = parameters["Wy"]
    by = parameters["by"]
    n_y = by.shape[0]
    ya = np.zeros((n_y, m, T_x))
    a0 = np.zeros((n_a, m))
    a0_ = np.zeros((n_a, m))
    x = X
    x_ = np.flip(x, axis = 2)

    a, c, caches = lstm_forward(x, a0, parameters, vocab_size, layer = 'later')
    a_, c_, caches_ = lstm_forward(x_, a0_, parameters, vocab_size, layer = 'later')

    ao = np.concatenate((a, a_), axis = 0)
    if activation == 'relu':
        for t in range(T_x):
            ya[:,:,t] = relu(np.dot(Wy, ao[:,:,t]) + by)
    if activation == 'softmax':
        for t in range(T_x):
            ya[:,:,t] = softmax(np.dot(Wy, ao[:,:,t]) + by)

    return ya, ao, caches, caches_

def lstm_cell_backward(da_next, dc_next, cache):

    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    n_x, m = xt.shape
    n_a, m = a_next.shape

    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next) * (1 - cct ** 2)
    dit = (dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.tanh(c_next) ** 2) * c_prev * da_next) * ft * (1 - ft)

    dWf = np.dot(dft, np.concatenate((a_prev, xt), axis = 0).T)
    dWi = np.dot(dit, np.concatenate((a_prev, xt), axis = 0).T)
    dWc = np.dot(dcct, np.concatenate((a_prev, xt), axis = 0).T)
    dWo = np.dot(dot, np.concatenate((a_prev, xt), axis = 0).T)
    dbf = np.sum(dft, axis = 1, keepdims = True)
    dbi = np.sum(dit, axis = 1, keepdims = True)
    dbc = np.sum(dcct, axis = 1, keepdims = True)
    dbo = np.sum(dot, axis = 1, keepdims = True)

    da_prev = np.dot(parameters['Wf'][:, : n_a].T, dft) + np.dot(parameters['Wi'][:, : n_a].T, dit) + np.dot(parameters['Wc'][:, : n_a].T, dcct) + np.dot(parameters['Wo'][:, : n_a].T, dot)
    dc_prev = dc_next * ft + dot * (1 - np.tanh(c_next) ** 2) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a : ].T, dft) + np.dot(parameters['Wi'][:, n_a : ].T, dit) + np.dot(parameters['Wc'][:, n_a : ].T, dcct) + np.dot(parameters['Wo'][:, n_a : ].T, dot)

    gradient = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradient

def lstm_backward(da, caches, gradients):

    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    dWf = gradients["dWf"]
    dWi = gradients["dWi"]
    dWc = gradients["dWc"]
    dWo = gradients["dWo"]
    dWy = gradients["dWy"]
    dbf = gradients["dbf"]
    dbi = gradients["dbi"]
    dbc = gradients["dbc"]
    dbo = gradients["dbo"]
    dby = gradients["dby"]
    dx = gradients["dx"]
    da0 = gradients["da0"]

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):

        gradient = lstm_cell_backward(da[:,:,t] + da_prevt, dc_prevt, caches[t])

        da_prevt = gradient["da_prev"]
        dc_prevt = gradient["dc_prev"]
        dx[:,:,t] = gradient["dxt"]
        dWf += gradient["dWf"]
        dWi += gradient["dWi"]
        dWc += gradient["dWc"]
        dWo += gradient["dWo"]
        dbf += gradient["dbf"]
        dbi += gradient["dbi"]
        dbc += gradient["dbc"]
        dbo += gradient["dbo"]

    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo, "dWy": dWy, "dby": dby}

    return gradients

def BiLSTM_backward_last(y_hat, Y, caches, caches_, ao, gradients):

    (caches_o, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches_o[0]

    n_a2, m, T_x = ao.shape
    n_x, m = x1.shape
    n_a = n_a2 // 2
    n_y = y_hat.shape[0]

    dZ = np.zeros(y_hat.shape)
    dao = np.zeros(ao.shape)
    Wy = parameters['Wy']

    for t in reversed(range(T_x)):
        dZ[:,:,t] = np.copy(y_hat[:,:,t])
        for j in range(m):
            dZ[int(Y[j, t]), j, t] -= 1
        dao[:,:,t] = np.dot(Wy.T, dZ[:,:,t])
        gradients["dWy"] += np.dot(dZ[:,:,t], ao[:,:,t].T)
        gradients["dby"] += np.sum(dZ[:,:,t], axis = 1, keepdims = True)

    da, da_ = dao[ : n_a, :, :], dao[n_a : , :, :]

    gradients = lstm_backward(da, caches, gradients)
    dx = gradients["dx"]
    da0 = gradients["da0"]
    gradients = lstm_backward(da_, caches_, gradients)
    dx_ = gradients["dx"]
    da0_ = gradients["da0"]

    dx_ = np.flip(dx_, axis = 2)

    gradients["dx"] = dx + dx_
    gradients["da0"] = da0
    gradients["da0_"] = da0_

    return gradients

def BiLSTM_backward_before(dya, caches, caches_, ao, gradients):

    (caches_o, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches_o[0]

    n_a2, m, T_x = ao.shape
    n_x, m = x1.shape
    n_a = n_a2 // 2
    n_y = dya.shape[0]

    dZ = np.zeros(dya.shape)
    Z = np.zeros(dya.shape)
    dao = np.zeros(ao.shape)
    Wy = parameters['Wy']
    by = parameters['by']

    for t in reversed(range(T_x)):
        Z[:,:,t] = np.dot(Wy, ao[:,:,t]) + by
        dZ[:,:,t] = ((Z[:,:,t] > 0).astype(int)) * dya[:,:,t]
        dao[:,:,t] = np.dot(Wy.T, dZ[:,:,t])
        gradients["dWy"] += np.dot(dZ[:,:,t], ao[:,:,t].T)
        gradients["dby"] += np.sum(dZ[:,:,t], axis = 1, keepdims = True)

    da, da_ = dao[ : n_a, :, :], dao[n_a : , :, :]

    gradients = lstm_backward(da, caches, gradients)
    dx = gradients["dx"]
    da0 = gradients["da0"]
    gradients = lstm_backward(da_, caches_, gradients)
    dx_ = gradients["dx"]
    da0_ = gradients["da0"]

    dx_ = np.flip(dx_, axis = 2)

    gradients["dx"] = dx + dx_
    gradients["da0"] = da0
    gradients["da0_"] = da0_

    return gradients

def get_loss(y_hat, Y):
    y = np.zeros(y_hat.shape)
    n_y, m, T_x = y_hat.shape
    for i in range(m):
        for j in range(T_x):
            y[int(Y[i, j]), i, j] = 1

    loss = -np.sum(y * np.log(y_hat))

    return loss

def update_parameters_with_Adam(parameters, gradients, v, s, t, lr = 0.01, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    dWf = gradients["dWf"]
    dWi = gradients["dWi"]
    dWc = gradients["dWc"]
    dWo = gradients["dWo"]
    dWy = gradients["dWy"]
    dbf = gradients["dbf"]
    dbi = gradients["dbi"]
    dbc = gradients["dbc"]
    dbo = gradients["dbo"]
    dby = gradients["dby"]

    vdWf = v["vdWf"]
    vdWi = v["vdWi"]
    vdWc = v["vdWc"]
    vdWo = v["vdWo"]
    vdWy = v["vdWy"]
    vdbf = v["vdbf"]
    vdbi = v["vdbi"]
    vdbc = v["vdbc"]
    vdbo = v["vdbo"]
    vdby = v["vdby"]

    sdWf = s["sdWf"]
    sdWi = s["sdWi"]
    sdWc = s["sdWc"]
    sdWo = s["sdWo"]
    sdWy = s["sdWy"]
    sdbf = s["sdbf"]
    sdbi = s["sdbi"]
    sdbc = s["sdbc"]
    sdbo = s["sdbo"]
    sdby = s["sdby"]

    vdWf = beta1 * vdWf + (1 - beta1) * dWf
    vdWi = beta1 * vdWi + (1 - beta1) * dWi
    vdWc = beta1 * vdWc + (1 - beta1) * dWc
    vdWo = beta1 * vdWo + (1 - beta1) * dWo
    vdWy = beta1 * vdWy + (1 - beta1) * dWy
    vdbf = beta1 * vdbf + (1 - beta1) * dbf
    vdbi = beta1 * vdbi + (1 - beta1) * dbi
    vdbc = beta1 * vdbc + (1 - beta1) * dbc
    vdbo = beta1 * vdbo + (1 - beta1) * dbo
    vdby = beta1 * vdby + (1 - beta1) * dby

    sdWf = beta2 * sdWf + (1 - beta2) * (dWf ** 2)
    sdWi = beta2 * sdWi + (1 - beta2) * (dWi ** 2)
    sdWc = beta2 * sdWc + (1 - beta2) * (dWc ** 2)
    sdWo = beta2 * sdWo + (1 - beta2) * (dWo ** 2)
    sdWy = beta2 * sdWy + (1 - beta2) * (dWy ** 2)
    sdbf = beta2 * sdbf + (1 - beta2) * (dbf ** 2)
    sdbi = beta2 * sdbi + (1 - beta2) * (dbi ** 2)
    sdbc = beta2 * sdbc + (1 - beta2) * (dbc ** 2)
    sdbo = beta2 * sdbo + (1 - beta2) * (dbo ** 2)
    sdby = beta2 * sdby + (1 - beta2) * (dby ** 2)

    vdWf_corrected = vdWf / (1 - beta1 ** t)
    vdWi_corrected = vdWi / (1 - beta1 ** t)
    vdWc_corrected = vdWc / (1 - beta1 ** t)
    vdWo_corrected = vdWo / (1 - beta1 ** t)
    vdWy_corrected = vdWy / (1 - beta1 ** t)
    vdbf_corrected = vdbf / (1 - beta1 ** t)
    vdbi_corrected = vdbi / (1 - beta1 ** t)
    vdbc_corrected = vdbc / (1 - beta1 ** t)
    vdbo_corrected = vdbo / (1 - beta1 ** t)
    vdby_corrected = vdby / (1 - beta1 ** t)

    sdWf_corrected = sdWf / (1 - beta2 ** t)
    sdWi_corrected = sdWi / (1 - beta2 ** t)
    sdWc_corrected = sdWc / (1 - beta2 ** t)
    sdWo_corrected = sdWo / (1 - beta2 ** t)
    sdWy_corrected = sdWy / (1 - beta2 ** t)
    sdbf_corrected = sdbf / (1 - beta2 ** t)
    sdbi_corrected = sdbi / (1 - beta2 ** t)
    sdbc_corrected = sdbc / (1 - beta2 ** t)
    sdbo_corrected = sdbo / (1 - beta2 ** t)
    sdby_corrected = sdby / (1 - beta2 ** t)

    Wf += -lr * vdWf_corrected / (np.sqrt(sdWf_corrected + epsilon))
    Wi += -lr * vdWi_corrected / (np.sqrt(sdWi_corrected + epsilon))
    Wc += -lr * vdWc_corrected / (np.sqrt(sdWc_corrected + epsilon))
    Wo += -lr * vdWo_corrected / (np.sqrt(sdWo_corrected + epsilon))
    bf += -lr * vdbf_corrected / (np.sqrt(sdbf_corrected + epsilon))
    bi += -lr * vdbi_corrected / (np.sqrt(sdbi_corrected + epsilon))
    bc += -lr * vdbc_corrected / (np.sqrt(sdbc_corrected + epsilon))
    bo += -lr * vdbo_corrected / (np.sqrt(sdbo_corrected + epsilon))
    by += -lr * vdby_corrected / (np.sqrt(sdby_corrected + epsilon))

    parameters = {"Wf": Wf,"bf": bf, "Wi": Wi,"bi": bi,
                "Wc": Wc,"bc": bc, "Wo": Wo,"bo": bo, "Wy": Wy, "by": by}
    v = {"vdWf": vdWf,"vdbf": vdbf, "vdWi": vdWi,"vdbi": vdbi,
        "vdWc": vdWc,"vdbc": vdbc, "vdWo": vdWo,"vdbo": vdbo, "vdWy": vdWy, "vdby": vdby}
    s = {"sdWf": sdWf,"sdbf": sdbf, "sdWi": sdWi,"sdbi": sdbi,
        "sdWc": sdWc,"sdbc": sdbc, "sdWo": sdWo,"sdbo": sdbo, "sdWy": sdWy, "sdby": sdby}

    return parameters, v, s

def sample(parameters1, parameters2, parameters3, parameters4, char_to_ind, sample_length, vocab_size):

    Wf1 = parameters1["Wf"]
    bf1 = parameters1["bf"]
    Wi1 = parameters1["Wi"]
    bi1 = parameters1["bi"]
    Wc1 = parameters1["Wc"]
    bc1 = parameters1["bc"]
    Wo1 = parameters1["Wo"]
    bo1 = parameters1["bo"]
    Wy1 = parameters1["Wy"]
    by1 = parameters1["by"]

    Wf2 = parameters2["Wf"]
    bf2 = parameters2["bf"]
    Wi2 = parameters2["Wi"]
    bi2 = parameters2["bi"]
    Wc2 = parameters2["Wc"]
    bc2 = parameters2["bc"]
    Wo2 = parameters2["Wo"]
    bo2 = parameters2["bo"]
    Wy2 = parameters2["Wy"]
    by2 = parameters2["by"]

    Wf3 = parameters3["Wf"]
    bf3 = parameters3["bf"]
    Wi3 = parameters3["Wi"]
    bi3 = parameters3["bi"]
    Wc3 = parameters3["Wc"]
    bc3 = parameters3["bc"]
    Wo3 = parameters3["Wo"]
    bo3 = parameters3["bo"]
    Wy3 = parameters3["Wy"]
    by3 = parameters3["by"]

    Wf4 = parameters4["Wf"]
    bf4 = parameters4["bf"]
    Wi4 = parameters4["Wi"]
    bi4 = parameters4["bi"]
    Wc4 = parameters4["Wc"]
    bc4 = parameters4["bc"]
    Wo4 = parameters4["Wo"]
    bo4 = parameters4["bo"]
    Wy4 = parameters4["Wy"]
    by4 = parameters4["by"]

    n_a1 = Wf1.shape[0]
    n_a2 = Wf2.shape[0]
    n_a3 = Wf3.shape[0]
    n_a4 = Wf4.shape[0]

    x = np.zeros((vocab_size, 1))
    a_prev1 = np.zeros((n_a1, 1))
    c_prev1 = np.zeros((n_a1, 1))
    a_prev2 = np.zeros((n_a2, 1))
    c_prev2 = np.zeros((n_a2, 1))
    a_prev3 = np.zeros((n_a3, 1))
    c_prev3 = np.zeros((n_a3, 1))
    a_prev4 = np.zeros((n_a4, 1))
    c_prev4 = np.zeros((n_a4, 1))

    indices = list()
    ind = -1

    for i in range(sample_length):

        concat1 = np.concatenate((a_prev1, x), axis = 0)
        ft1 = sigmoid(np.dot(Wf1, concat1) + bf1)
        it1 = sigmoid(np.dot(Wi1, concat1) + bi1)
        cct1 = np.tanh(np.dot(Wc1, concat1) + bc1)
        c_next1 = (ft1 * c_prev1) + (it1 * cct1)
        ot1 = sigmoid(np.dot(Wo1, concat1) + bo1)
        a1 = ot1 * np.tanh(c_next1)
        ya1 = relu(np.dot(Wy1[:, : n_a1], a1) + by1)

        concat2 = np.concatenate((a_prev2, ya1), axis = 0)
        ft2 = sigmoid(np.dot(Wf2, concat2) + bf2)
        it2 = sigmoid(np.dot(Wi2, concat2) + bi2)
        cct2 = np.tanh(np.dot(Wc2, concat2) + bc2)
        c_next2 = (ft2 * c_prev2) + (it2 * cct2)
        ot2 = sigmoid(np.dot(Wo2, concat2) + bo2)
        a2 = ot2 * np.tanh(c_next2)
        ya2 = relu(np.dot(Wy2[:, : n_a2], a2) + by2)

        concat3 = np.concatenate((a_prev3, ya2), axis = 0)
        ft3 = sigmoid(np.dot(Wf3, concat3) + bf3)
        it3 = sigmoid(np.dot(Wi3, concat3) + bi3)
        cct3 = np.tanh(np.dot(Wc3, concat3) + bc3)
        c_next3 = (ft3 * c_prev3) + (it3 * cct3)
        ot3 = sigmoid(np.dot(Wo3, concat3) + bo3)
        a3 = ot3 * np.tanh(c_next3)
        ya3 = relu(np.dot(Wy3[:, : n_a3], a3) + by3)

        concat4 = np.concatenate((a_prev4, x4), axis = 0)
        ft4 = sigmoid(np.dot(Wf4, concat4) + bf4)
        it4 = sigmoid(np.dot(Wi4, concat4) + bi4)
        cct4 = np.tanh(np.dot(Wc4, concat4) + bc4)
        c_next4 = (ft4 * c_prev4) + (it4 * cct4)
        ot4 = sigmoid(np.dot(Wo4, concat4) + bo4)
        a4 = ot4 * np.tanh(c_next4)
        y4 = softmax(np.dot(Wy4[:, : n_a4], a4) + by4)

        ind = np.random.choice(range(vocab_size), p = np.ravel(y4))
        indices.append(ind)

        x = np.zeros((vocab_size, 1))
        x[ind] = 1

        a_prev1 = a1
        c_prev1 = c_next1
        a_prev2 = a2
        c_prev2 = c_next2
        a_prev3 = a3
        c_prev3 = c_next3
        a_prev4 = a4
        c_prev4 = c_next4

    return indices

def optimize(X, Y, parameters1, parameters2, parameters3, parameters4, vs, ss, vocab_size, t, learning_rate = 0.01):

    m, T_x = X.shape
    n_x1 = vocab_size

    ya1, ao1, caches1, caches1_ = BiLSTM_forward_first(X, parameters1, vocab_size, activation = 'relu')
    ya2, ao2, caches2, caches2_ = BiLSTM_forward_later(ya1, parameters2, vocab_size, activation = 'relu')
    ya3, ao3, caches3, caches3_ = BiLSTM_forward_later(ya2, parameters3, vocab_size, activation = 'relu')
    ya4, ao4, caches4, caches4_ = BiLSTM_forward_later(ya3, parameters4, vocab_size, activation = 'softmax')

    loss = get_loss(ya4, Y)

    n_x2 = ya1.shape[0]
    n_x3 = ya2.shape[0]
    n_x4 = ya3.shape[0]
    n_y1 = ya1.shape[0]
    n_y2 = ya2.shape[0]
    n_y3 = ya3.shape[0]
    n_y4 = ya4.shape[0]
    n_a1 = parameters1["Wf"].shape[0]
    n_a2 = parameters2["Wf"].shape[0]
    n_a3 = parameters3["Wf"].shape[0]
    n_a4 = parameters4["Wf"].shape[0]

    gradients1 = create_gradients(n_x1, n_a1, n_y1, m, T_x)
    gradients2 = create_gradients(n_x2, n_a2, n_y2, m, T_x)
    gradients3 = create_gradients(n_x3, n_a3, n_y3, m, T_x)
    gradients4 = create_gradients(n_x4, n_a4, n_y4, m, T_x)

    v1, s1 = vs["v1"], ss["s1"]
    v2, s2 = vs["v2"], ss["s2"]
    v3, s3 = vs["v3"], ss["s3"]
    v4, s4 = vs["v4"], ss["s4"]

    gradients4 = BiLSTM_backward_last(ya4, Y, caches4, caches4_, ao4, gradients4)
    gradients3 = BiLSTM_backward_before(gradients4["dx"], caches3, caches3_, ao3, gradients3)
    gradients2 = BiLSTM_backward_before(gradients3["dx"], caches2, caches2_, ao2, gradients2)
    gradients1 = BiLSTM_backward_before(gradients2["dx"], caches1, caches1_, ao1, gradients1)

    gradients1 = clip(gradients1, 10)
    gradients2 = clip(gradients2, 10)
    gradients3 = clip(gradients3, 10)
    gradients4 = clip(gradients4, 10)

    parameters1, v1, s1 = update_parameters_with_Adam(parameters1, gradients1, v1, s1, t, lr = learning_rate)
    parameters2, v2, s2 = update_parameters_with_Adam(parameters2, gradients2, v2, s2, t, lr = learning_rate)
    parameters3, v3, s3 = update_parameters_with_Adam(parameters3, gradients3, v3, s3, t, lr = learning_rate)
    parameters4, v4, s4 = update_parameters_with_Adam(parameters4, gradients4, v4, s4, t, lr = learning_rate)

    vs["v1"], ss["s1"] = v1, s1
    vs["v2"], ss["s2"] = v2, s2
    vs["v3"], ss["s3"] = v3, s3
    vs["v4"], ss["s4"] = v4, s4

    return loss, parameters1, parameters2, parameters3, parameters4, ya4, vs, ss

def create_datasets(data, T_x, data_size, char_to_ind):
    X = list()
    Y = list()

    for i in range(data_size - T_x - 1):
        x = list(data[i: i + T_x])
        x = [char_to_ind[c] for c in x]
        X.append(x)
        y = list(data[i + 1: i + T_x + 1])
        y = [char_to_ind[c] for c in y]
        Y.append(y)

    # test_size = 50000
    # train_size = len(X) - test_size

    random.shuffle(X)
    random.shuffle(Y)

    # X_train = X[ : train_size]
    # Y_train = Y[ : train_size]
    # X_test = X[train_size : train_size + test_size]
    # Y_test = Y[train_size : train_size + test_size]

    # return X_train, Y_train, X_test, Y_test
    return X, Y

def get_accuracy(y_hat, Y):
    m, T_x = Y.shape
    y_h = np.zeros(Y.shape)
    y_max = np.amax(y_hat, axis = 0)
    for i in range(m):
        for j in range(T_x):
            y_h[i, j] = np.where(y_hat[:, i, j] == y_max[i, j])[0][0]

    tot = m * T_x
    sum = 0
    for i in range(m):
        for j in range(T_x):
            if (y_h[i, j] == Y[i, j]):
                sum += 1
    accuracy = sum * 100 / tot

    return accuracy

def model(vocab_size, n_a1, n_a2, n_a3, n_a4, X, Y, n_y1, n_y2, n_y3, num_epochs = 10, learning_rate = 0.01, m = 1000):

    n_x1, n_y4 = vocab_size, vocab_size
    n_x2, n_x3, n_x4 = n_y1, n_y2, n_y3

    vs = dict()
    ss= dict()

    parameters1 = initialize_parameters(n_x1, n_a1, n_y1)
    parameters2 = initialize_parameters(n_x2, n_a2, n_y2)
    parameters3 = initialize_parameters(n_x3, n_a3, n_y3)
    parameters4 = initialize_parameters(n_x4, n_a4, n_y4)

    vs["v1"], ss["s1"] = initialize_v_and_s(parameters1)
    vs["v2"], ss["s2"] = initialize_v_and_s(parameters2)
    vs["v3"], ss["s3"] = initialize_v_and_s(parameters3)
    vs["v4"], ss["s4"] = initialize_v_and_s(parameters4)

    x = list()
    y = list()
    l = len(X) // m
    for i in range(l):
        x.append(np.array(X[(i * m) : ((i + 1) * m)], dtype = np.int8))
        y.append(np.array(Y[(i * m) : ((i + 1) * m)], dtype = np.int8))
    x.append(np.array(X[(l * m) : ], dtype = np.int8))
    y.append(np.array(Y[(l * m) : ], dtype = np.int8))

    t = 0
    for i in range(num_epochs):
        random.shuffle(x)
        random.shuffle(y)
        for j in range(l + 1):
            t += 1
            loss, parameters1, parameters2, parameters3, parameters4, y_hat, vs, ss = optimize(x[j], y[j], parameters1, parameters2, parameters3, parameters4, vs, ss, vocab_size, t, learning_rate = learning_rate)
            accuracy = get_accuracy(y_hat, y[j])
            if t % 100 == 0:
                print('Iteration = ', t, '  Loss = ', loss, '  Accuracy = ', accuracy )

    return parameters1, parameters2, parameters3, parameters4, loss, accuracy

T_x = 100
sample_length = 1200
# X_train, Y_train, X_test, Y_test = create_datasets(data, T_x, data_size, char_to_ind)
X_train, Y_train = create_datasets(data, T_x, data_size, char_to_ind)
p1, p2, p3, p4, train_loss, train_accuracy = model(vocab_size, 150, 100, 50, 25, X_train, Y_train, 80, 40, 20, num_epochs = 2, learning_rate = 0.01, m = 500)
# ya1, ao1, caches1, caches1_ = BiLSTM_forward_first(X_test, p1, vocab_size, activation = 'relu')
# ya2, ao2, caches2, caches2_ = BiLSTM_forward_later(ya1, p2, vocab_size, activation = 'relu')
# ya3, ao3, caches3, caches3_ = BiLSTM_forward_later(ya2, p3, vocab_size, activation = 'relu')
# ya4, ao4, caches4, caches4_ = BiLSTM_forward_later(ya3, p4, vocab_size, activation = 'softmax')
# test_accuracy = get_accuracy(ya4, np.array(Y_test))

print('Train Accuracy = ', train_accuracy)
# print('Test Accuracy = ', test_accuracy)

sampled_indices = sample(p1, p2, p3, p4, char_to_ind, sample_length, vocab_size)
text = print_sample(sampled_indices, ind_to_char)
print(text)
