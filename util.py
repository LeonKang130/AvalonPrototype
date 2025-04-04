import torch
import numpy as np
from typing import Tuple
import OpenEXR, Imath

def sample_sphere_uniform(n: int, device: torch.device) -> torch.Tensor:
    phi = 2.0 * torch.pi * torch.rand(n, dtype=torch.float32, device=device)
    cos_theta = 1.0 - 2.0 * torch.rand(n, dtype=torch.float32, device=device)
    sin_theta = torch.sqrt(1.0 - torch.square(cos_theta))
    return torch.stack((sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), cos_theta), dim=-1)

def sample_hemisphere_uniform(n: int, device: torch.device) -> torch.Tensor:
    phi = 2.0 * torch.pi * torch.rand(n, dtype=torch.float32, device=device)
    cos_theta = torch.rand(n, dtype=torch.float32, device=device)
    sin_theta = torch.sqrt(1.0 - torch.square(cos_theta))
    return torch.stack((sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), cos_theta), dim=-1)

def sample_hemisphere_cosine(n: int, device: torch.device) -> torch.Tensor:
    phi = 2.0 * torch.pi * torch.rand(n, dtype=torch.float32, device=device)
    cos_theta = torch.sqrt(torch.rand(n, dtype=torch.float32, device=device))
    sin_theta = torch.sqrt(1.0 - torch.square(cos_theta))
    return torch.stack((torch.cos(phi) * sin_theta, torch.sin(phi) * sin_theta, cos_theta), dim=-1)

def make_onb(ns: torch.Tensor) -> torch.Tensor:
    a, b = torch.tensor([0.0, 1.0, 0.0], dtype=ns.dtype, device=ns.device), torch.tensor([1.0, 0.0, 0.0], dtype=ns.dtype, device=ns.device)
    ts = torch.where(torch.abs(ns[..., 0:1]) > 0.1, a, b)
    ts -= ns * (ns * ts).sum(dim=-1).unsqueeze(-1)
    ts /= (torch.linalg.norm(ts, dim=-1) + torch.finfo(torch.float32).eps).unsqueeze(-1)
    bs = torch.linalg.cross(ns, ts)
    return torch.stack((ts, bs, ns), dim=-2)

def sample_onb_uniform(n: int, device: torch.device) -> torch.Tensor:
    onb = make_onb(sample_sphere_uniform(n, device))
    ksi = torch.rand(n, 1, dtype=torch.float32, device=device)
    cos_ksi, sin_ksi = torch.cos(ksi), torch.sin(ksi)
    return torch.stack([
        cos_ksi * onb[..., 0] + sin_ksi * onb[..., 1],
        cos_ksi * onb[..., 1] - sin_ksi * onb[..., 0],
        onb[..., 2]
    ], dim=-2)

def evaluate_sf(n: int, device: torch.device) -> torch.Tensor:
    phi = 2.0 * torch.pi * torch.rand(n, dtype=torch.float32, device=device)
    cos_theta = 1.0 - (2.0 * torch.arange(n, dtype=torch.float32, device=device) + 1.0) / n
    sin_theta = torch.sqrt(1.0 - torch.square(cos_theta))
    return torch.stack((torch.cos(phi) * sin_theta, torch.sin(phi) * sin_theta, cos_theta), dim=-1)

def evaluate_sh(ws: torch.Tensor) -> torch.Tensor:
    ks = 0.28209479177387814, 0.4886025119029199
    num_spherical_harmonic_basis = 4
    sh = torch.empty(ws.shape[:-1] + (num_spherical_harmonic_basis,), dtype=ws.dtype, device=ws.device)
    sh[..., 0] = ks[0]
    sh[..., 1] = -ks[1] * ws[..., 1]
    sh[..., 2] = ks[1] * ws[..., 2]
    sh[..., 3] = -ks[1] * ws[..., 0]
    return sh

def fit_sh_coefficients(ws: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(evaluate_sh(ws), ys, rcond=None).solution

def evaluate_octahedron(uv: np.ndarray) -> np.ndarray:
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

def generate_visualization_ws(resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    i = np.arange(resolution * resolution)
    xy = np.hstack((np.tile(i, resolution * resolution)[..., None], np.repeat(i, resolution * resolution)[..., None]))
    xy_wi, xy_wo = np.divmod(xy, resolution)
    return evaluate_octahedron((xy_wi + 0.5) / resolution), evaluate_octahedron((xy_wo + 0.5) / resolution)

def save_exr_image(image: torch.Tensor, filename: str):
    exr_data = [image[..., i].cpu().numpy().astype(np.float32).tobytes() for i in range(image.shape[-1])]
    header = OpenEXR.Header(*image.shape[:2])
    channels = ["R", "G", "B", "A"][:len(exr_data)]
    header["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in channels}
    exr_file = OpenEXR.OutputFile(filename, header)
    exr_file.writePixels(dict(zip(channels, exr_data)))
    exr_file.close()
