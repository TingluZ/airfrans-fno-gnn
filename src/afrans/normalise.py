import numpy as np

class RunningStats:
    """
    Mean/std over batches that never all exist at once

    Sums are shifted by a constant before accumulating:
    several AirfRANS columns (u_xin, u_yin) are constant within a 
    simulation, and the naive E[x^2] - E[x]^2 form losses all precision
    when std << |mean|
    """

    def __init__(self):
        self.shift=None
        self.n = 0

    def update(self, a):
        a = np.array(a, dtype=np.float64)
        if self.shift is None:
            self.shift = a.mean(axis=0)
            self.s1 = np.zeros_like(self.shift)
            self.s2 = np.zeros_like(self.shift)
        d = a - self.shift
        self.n += a.shape[0]
        self.s1 += d.sum(axis=0)
        self.s2 += (d*d).sum(axis=0)

    def result(self):
        mean = self.shift + self.s1 / self.n
        var = (self.s2 - self.s1**2 / self.n) / self.n
        return mean, np.sqrt(np.maximum(var, 0.0))

class Normaliser:

    def __init__(self):
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None


    def fit(self, feature_list, target_list):
        feature = np.concatenate(feature_list, axis=0)
        target = np.concatenate(target_list, axis=0)

        self.feature_mean, self.feature_std = feature.mean(axis=0), feature.std(axis=0) + 1e-8
        self.target_mean, self.target_std = target.mean(axis=0), target.std(axis=0) + 1e-8

    def transform_feature(self, feature):
        return (feature - self.feature_mean)/self.feature_std
    
    def transform_target(self, target):
        return (target - self.target_mean)/self.target_std
    
    def inverse_target(self, target):
        return target * self.target_std + self.target_mean
    
    def save(self, path):
        np.savez(path, 
                 x_mean = self.feature_mean,
                 x_std = self.feature_std,
                 y_mean = self.target_mean,
                 y_std = self.target_std)
        
    def load(self, path):
        d = np.load(path)
        self.feature_mean, self.feature_std = d["x_mean"], d["x_std"]
        self.target_mean, self.target_std = d["y_mean"], d["y_std"]

    def fit_from_stats(self, f_mean, f_std, t_mean, t_std, eps=1e-8):
        self.feature_mean, self.feature_std = f_mean, f_std + eps
        self.target_mean, self.target_std = t_mean, t_std + eps