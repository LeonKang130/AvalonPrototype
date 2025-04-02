import numpy as np

class DisneyBSDF(object):
    def __init__(self, metallic: float, roughness: float, ior: float, base_color: np.ndarray):
        self.metallic = metallic
        self.roughness = max(roughness, 0.1)
        self.reflectance = ((1.0 - ior) / (1.0 + ior)) ** 2
        self.base_color = np.array(base_color)

    def _lambda(self, w: np.ndarray) -> np.ndarray:
        cos2theta = w[..., 2] * w[..., 2] + np.finfo(float).eps
        tan2theta = np.where(cos2theta < 1e-6, np.zeros_like(cos2theta), (1.0 - cos2theta) / cos2theta)
        return (-1.0 + np.sqrt(1.0 + self.roughness * self.roughness * tan2theta)) / 2.0

    def _d(self, w: np.ndarray) -> np.ndarray:
        nominator = self.roughness * self.roughness * np.where(w[..., 2] > 0.0, 1.0, 0.0)
        denominator = np.pi * np.square(1.0 + (self.roughness * self.roughness - 1.0) * w[..., 2] * w[..., 2])
        return nominator / (denominator + np.finfo(float).eps)

    def _g(self, wi: np.ndarray, wo: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + self._lambda(wi) + self._lambda(wo))

    def evaluate(self, wo: np.ndarray, wi: np.ndarray) -> np.ndarray:
        wh = wi + wo
        wh /= np.linalg.norm(wh, axis=-1)[..., None] + np.finfo(float).eps
        fc = np.power(1.0 - (wi * wh).sum(axis=-1), 5.0)[..., None]
        f0 = self.metallic * self.base_color + (1.0 - self.metallic) * self.reflectance
        f = f0 + (1.0 - f0) * fc
        specular_bsdf = (self._d(wh) * self._g(wi, wo))[..., None] * f / (4.0 * np.abs(wi[..., 2]) * np.abs(wo[..., 2]) + np.finfo(float).eps)[..., None]
        diffuse_bsdf = ((1.0 - self.metallic) * (1.0 - self.reflectance) * self.base_color) * (1.0 - fc) * np.where(np.minimum(wi[..., 2], wo[..., 2]) < np.finfo(float).eps, 0.0, 1.0)[..., None] / np.pi
        return specular_bsdf + diffuse_bsdf