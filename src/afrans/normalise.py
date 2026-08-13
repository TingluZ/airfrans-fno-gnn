import numpy as np

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