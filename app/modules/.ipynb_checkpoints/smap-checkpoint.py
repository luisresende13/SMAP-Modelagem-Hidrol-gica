import numpy as np

def ModeloSmapDiario(Ad, Str, K2t, Crec, Ai, Capc, Kkt, Ep, Pr, Pcof, Tuin, Ebin, Supin, kep, H, K1t, K3t):
    ndias = len(Pr) - 1

    Rsolo = [0] * (ndias + 1)
    Rsub = [0] * (ndias + 1)
    Rsup = [0] * (ndias + 1)
    Rsup2 = [0] * (ndias + 1)
    P = [0] * (ndias + 1)
    Es = [0] * (ndias + 1)
    Er = [0] * (ndias + 1)
    Rec = [0] * (ndias + 1)
    Ed = [0] * (ndias + 1)
    Emarg = [0] * (ndias + 1)
    Ed2 = [0] * (ndias + 1)
    Eb = [0] * (ndias + 1)
    Q = [0] * (ndias + 1)
    
    Rsolo[0] = Tuin / 100 * Str
    Rsub[0] = Ebin / (1 - 0.5 ** (1 / Kkt)) / Ad * 86.4
    Rsup[0] = Supin / (1 - 0.5 ** (1 / K2t)) / Ad * 86.4
    Rsup2[0] = 0.0

    for i in range(1, ndias + 1):
        P[i] = Pr[i] * Pcof
        Tu = Rsolo[i - 1] / Str
        
        if P[i] > Ai:
            Es[i] = (P[i] - Ai) ** 2 / (P[i] - Ai + Str - Rsolo[i - 1])
        else:
            Es[i] = 0
        
        if (P[i] - Es[i]) > Ep[i] * kep:
            Er[i] = Ep[i]
        else:
            Er[i] = (P[i] - Es[i]) + (Ep[i] - (P[i] - Es[i])) * Tu
        
        if Rsolo[i - 1] > (Capc / 100 * Str):
            Rec[i] = Crec / 100 * Tu * (Rsolo[i - 1] - (Capc / 100 * Str))
        else:
            Rec[i] = 0
        
        Rsolo[i] = Rsolo[i - 1] + P[i] - Es[i] - Er[i] - Rec[i]
        
        if Rsolo[i] > Str:
            Es[i] += Rsolo[i] - Str
            Rsolo[i] = Str
        
        Ed[i] = Rsup[i - 1] * (1 - 0.5 ** (1 / K2t))
        Rsup[i] = Rsup[i - 1] + Es[i] - Ed[i]
        
        if Rsup[i] > H:
            Emarg[i] = (Rsup[i] - H) * (1 - 0.5 ** (1 / K1t))
            Rsup[i] = H
        else:
            Emarg[i] = 0
        
        Ed2[i] = Rsup2[i - 1] * (1 - 0.5 ** (1 / K3t))
        Rsup2[i] = Rsup2[i - 1] + Emarg[i]
        
        Eb[i] = Rsub[i - 1] * (1 - 0.5 ** (1 / Kkt))
        Rsub[i] = Rsub[i - 1] + Rec[i] - Eb[i]
        # Rsup[i] = np.clip(Rsup[i - 1] + Es[i] - Ed[i], -1e24, 1e24)
        
        Q[i] = (Ed[i] + Eb[i] + Ed2[i]) * Ad / 86.4
        # Q[i] = np.clip((Ed[i] + Eb[i] + Ed2[i]) * Ad / 86.4, -1e24, 1e24)  # Example clamping

    # Create a dictionary with the resulting lists
    result = {
        'Rsolo': Rsolo,
        'Rsub': Rsub,
        'Rsup': Rsup,
        'Rsup2': Rsup2,
        'P': P,
        'Es': Es,
        'Er': Er,
        'Rec': Rec,
        'Ed': Ed,
        'Emarg': Emarg,
        'Ed2': Ed2,
        'Eb': Eb,
        'Q': Q
    }
    
    return result

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class SmapModel(BaseEstimator, RegressorMixin):
    def __init__(self, Ad, Str, K2t, Crec, Ai, Capc, Kkt, kep, H, K1t, K3t, Pcof, Tuin, Ebin, Supin):
        self.Ad = Ad
        self.Str = Str
        self.K2t = K2t
        self.Crec = Crec
        self.Ai = Ai
        self.Capc = Capc
        self.Kkt = Kkt
        self.kep = kep
        self.H = H
        self.K1t = K1t
        self.K3t = K3t
        self.Pcof = Pcof
        self.Tuin = Tuin
        self.Ebin = Ebin
        self.Supin = Supin
    
    def fit(self, X, y=None):
        # Fit method does nothing as this is a deterministic model
        return self
    
    def predict(self, X):
        # Assume X is a DataFrame with columns 'Ep' and 'Pr'
        Ep = X['Ep'].values
        Pr = X['Pr'].values
        
        result = ModeloSmapDiario(
            self.Ad, self.Str, self.K2t, self.Crec, self.Ai, self.Capc,
            self.Kkt, Ep, Pr, self.Pcof, self.Tuin, self.Ebin, self.Supin, self.kep, self.H, self.K1t, self.K3t
        )
        
        # Return the predicted values, i.e., the 'Q' from the result dictionary
        return np.array(result['Q'])  # Skip the first element as it's usually an initialization value
        # return np.array(result['Q'][1:])  # Skip the first element as it's usually an initialization value

# Example usage:
# from sklearn.model_selection import GridSearchCV

# Define the parameter grid
# param_grid = {
#     'Ad': [10, 20],
#     'Str': [50, 100],
#     'K2t': [5, 10],
#     'Crec': [0.1, 0.2],
#     # Add other parameters as needed
# }

# model = SmapModel(ndias=100, Ad=10, Str=50, K2t=5, Crec=0.1, Ai=5, Capc=30, Kkt=5, kep=1, H=10, K1t=1, K3t=1, Pcof=1, Tuin=10, Ebin=5, Supin=3)
# grid_search = GridSearchCV(model, param_grid, cv=3)
# grid_search.fit(X_train, y_train)
