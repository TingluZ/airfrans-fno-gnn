import torch
from torch.utils.data import Dataset
from scipy.spatial import Delaunay
from afrans.interpolate import point_to_grid

class AirfRANSGridDataset(Dataset):
    def __init__(
            self, 
            dataset, 
            indices, 
            normaliser, 
            grid_res_x = GRID_RES_X, 
            grid_res_y = GRID_RES_Y):
        
        self.samples = []
        print(f"Pre-interpolating {len(indices)} sim on to {grid_res_x}x{grid_res_y}...")
        gx, gy = np.meshgrid(
                np.linspace(X_MIN, X_MAX, grid_res_x),
                np.linspace(Y_MIN, Y_MAX, grid_res_y)
            )
        
        for i in indices:
            sim = dataset[i]
            pos = sim[:, 0:2]
            tri = Delaunay(pos)
            x_norm = normaliser.transform_feature(sim[:, 0:7])
            y_norm = normaliser.transform_target(sim[:, 7:11])
            x_grid = point_to_grid(x_norm, tri, gx, gy)
            y_grid = point_to_grid(y_norm, tri, gx, gy)

            self.samples.append({
                'x_grid':torch.from_numpy(x_grid),
                'y_grid':torch.from_numpy(y_grid),
                'pos':torch.from_numpy(pos.astype(np.float32)),
                'y_pts':torch.from_numpy(sim[:, 7:11].astype(np.float32))
                })

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]