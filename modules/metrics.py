import numpy as np

def nash_sutcliffe_efficacy(Q_obs, Q_pred):
    Q_obs = np.array(Q_obs)
    Q_pred = np.array(Q_pred)
    mean_Q_obs = np.mean(Q_obs)
    
    numerator = np.sum((Q_obs - mean_Q_obs) ** 2) - np.sum((Q_obs - Q_pred) ** 2)
    denominator = np.sum((Q_obs - mean_Q_obs) ** 2)
    
    nse = numerator / denominator
    return nse

def relative_error_coefficient(Q_obs, Q_pred):
    Q_obs = np.array(Q_obs)
    Q_pred = np.array(Q_pred)
    n = len(Q_obs)
    
    numerator = np.sum(np.abs(Q_pred - Q_obs) / Q_obs)
    
    re = 1 - numerator / n
    return re

import numpy as np

def correlation_coefficient(observed, simulated):
    mx = np.mean(observed)
    m_sim = np.mean(simulated)
    sx = np.std(observed, ddof=1)
    s_sim = np.std(simulated, ddof=1)
    
    R = np.mean(((observed - mx) / sx) * ((simulated - m_sim) / s_sim))
    return R

def mean_error(observed, simulated):
    EM = np.mean(observed - simulated)
    return EM

def mean_abs_error(observed, simulated):
    EM = np.mean(np.abs(observed - simulated))
    return EM

def normalized_rmse(observed, simulated):
    sx = np.std(observed, ddof=1)
    ERM = np.mean(((observed - simulated) / sx) ** 2)
    return ERM

def rmse(observed, simulated):
    EMQ = np.sqrt(np.mean((observed - simulated) ** 2))
    return EMQ
