import numpy as np

class SGGX(object):
    def __init__(self, eigenvectors: np.ndarray, eigenvalues: np.ndarray):
        self.s = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        self.inv_s = np.linalg.inv(self.s)
        sigma_x = np.linalg.norm(self.s[0])
        sigma_y = np.linalg.norm(self.s[1])
        sigma_z = np.linalg.norm(self.s[2])
        r_xy = self.s[0, 1] / (sigma_x * sigma_y)
        r_xz = self.s[0, 2] / (sigma_x * sigma_z)
        r_yz = self.s[1, 2] / (sigma_y * sigma_z)
        self.features = np.array([sigma_x, sigma_y, sigma_z, r_xy, r_xz, r_yz])

    def evaluate(self, w: np.ndarray) -> np.ndarray:
        if w.shape[-1] != 3:
            print("Input directions must be an array of 3D vectors")
            exit(1)
        return 1.0 / (np.pi * np.square((w * (w @ self.inv_s)).sum(axis=-1)))
