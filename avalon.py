from sggx import SGGX
from util import evaluate_sf, evaluate_sh, fit_sh_coefficients, sample_onb_uniform
from disney import DisneyBSDF
import numpy as np

class Avalon(object):
    def __init__(self):
        self.sggx = SGGX(sample_onb_uniform(), 1.0 - np.random.rand(3))
        num_sf_samples = 4
        ws = evaluate_sf(num_sf_samples, True)
        ys = np.random.rand(num_sf_samples, 6)
        self.sh_coefficients = fit_sh_coefficients(ws, ys)
        self.feature = np.hstack((self.sggx.features, self.sh_coefficients.flatten()))

    def evaluate(self, wi: np.ndarray, wo: np.ndarray, num_samples: int = 256):
        nominator = np.zeros(wi.shape[:-1] + (3,))
        denominator = np.finfo(float).eps
        # TODO: importance sampling
        for i in range(num_samples):
            onb = sample_onb_uniform()
            n = onb[2]
            bsdf_params = np.clip(evaluate_sh(n).dot(self.sh_coefficients), 0.0, 1.0)
            bsdf = DisneyBSDF(bsdf_params[0], bsdf_params[1], bsdf_params[2] + 1.0, bsdf_params[3:])
            wi_local = wi @ onb.T
            wo_local = wo @ onb.T
            bsdf_value = bsdf.evaluate(wo_local, wi_local)
            ndf_value = self.sggx.evaluate(n)
            nominator += bsdf_value * (ndf_value * np.maximum(wi_local[..., 2], 0.0) * np.maximum(wo_local[..., 2], 0.0))[..., None]
            denominator += ndf_value * np.maximum(wo_local[..., 2], 0.0)
        return nominator / denominator[..., None]