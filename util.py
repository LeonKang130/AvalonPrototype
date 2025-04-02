import numpy as np
import numba
import math

@numba.njit
def sample_sphere_uniform(n: int):
    phi = 2.0 * np.pi * np.random.rand(n, 1)
    cos_theta = 1.0 - 2.0 * np.random.rand(n, 1)
    sin_theta = np.sqrt(1.0 - np.square(cos_theta))
    return np.hstack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

@numba.njit
def sample_hemisphere_uniform(n: int):
    phi = 2.0 * np.pi * np.random.rand(n, 1)
    cos_theta = np.random.rand(n, 1)
    sin_theta = np.sqrt(1.0 - np.square(cos_theta))
    return np.hstack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

def sample_onb_uniform():
    phi = 2.0 * np.pi * np.random.rand(1)
    cos_theta = np.random.rand(1)
    sin_theta = np.sqrt(1.0 - np.square(cos_theta))
    n = np.hstack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    t = np.where((np.abs(n[0]) > 0.1)[..., None], np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    t = np.cross(n, t)
    t /= np.linalg.norm(t)
    b = np.cross(n, t)
    ksi = 2.0 * np.pi * np.random.rand(1)
    cos_ksi = np.cos(ksi)
    sin_ksi = np.sin(ksi)
    return np.array([cos_ksi * t + sin_ksi * b, cos_ksi * b - sin_ksi * t, n])

def evaluate_sf(n: int, rotate: bool = False) -> np.ndarray:
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    i = np.arange(n, dtype=np.float32)[..., None]
    phi = 2.0 * np.pi * np.modf(i * golden_ratio)[0]
    cos_theta = 1.0 - (2.0 * i + 1.0) / n
    sin_theta = np.sqrt(1.0 - np.square(cos_theta))
    ws = np.hstack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))
    if rotate:
        onb = sample_onb_uniform()
        return ws @ onb.T
    return np.hstack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))

@numba.njit
def evaluate_sh(w: np.ndarray) -> np.ndarray:
    ks = 0.28209479177387814, 0.4886025119029199
    num_spherical_harmonic_basis = 4
    """
    if w.shape[-1] != 3:
       print("Input directions must be an array of 3D vectors")
       exit(1)
    """
    sh = np.empty(w.shape[:-1] + (num_spherical_harmonic_basis,))
    sh[..., 0] = ks[0]
    sh[..., 1] = -ks[1] * w[..., 1]
    sh[..., 2] = ks[1] * w[..., 2]
    sh[..., 3] = -ks[1] * w[..., 0]
    return sh

def fit_sh_coefficients(ws: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    if ws.shape[-1] != 3:
        print("Input directions must be an array of 3D vectors")
        exit(1)
    """
    a = evaluate_sh(ws)
    return np.linalg.lstsq(a, ys, rcond=None)[0]

@numba.njit
def evaluate_octahedron(uv: np.ndarray) -> np.ndarray:
    """
    if uv.shape[-1] != 2 or uv.ndim == 1:
        print("Input directions must be an array of 2D vectors")
        exit(1)
    """
    uv = 2.0 * uv - 1.0
    uvp = np.abs(uv)
    signed_distance = 1.0 - uvp[..., 0] - uvp[..., 1]
    d = np.abs(signed_distance)
    r = 1.0 - d
    phi = np.where(r == 0, np.ones(uv.shape[:-1]), (uvp[..., 1] - uvp[..., 0]) / r + 1.0) * np.pi / 4.0
    z = np.copysign(1.0 - r * r, signed_distance)
    cos_phi = np.copysign(np.cos(phi), uv[..., 0])
    sin_phi = np.copysign(np.sin(phi), uv[..., 1])
    return np.stack((r * cos_phi, r * sin_phi, z), axis=-1)
