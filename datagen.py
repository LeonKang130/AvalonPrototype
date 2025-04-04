import torch

from util import *
from typing import NamedTuple

class SGGX(NamedTuple):
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    @property
    def feature(self):
        s = self.eigenvectors.T @ torch.diag(self.eigenvalues) @ self.eigenvectors
        sigma = torch.sqrt(torch.tensor([s[0, 0], s[1, 1], s[2, 2]], dtype=self.eigenvectors.dtype, device=self.eigenvectors.device))
        r = torch.tensor([s[0, 1] / sigma[0] / sigma[1], s[0, 2] / sigma[0] / sigma[2], s[1, 2] / sigma[1] / sigma[2]], dtype=self.eigenvectors.dtype, device=self.eigenvectors.device)
        return torch.hstack((sigma, r))

class Aggregate(NamedTuple):
    ndf: SGGX
    bsdf_sh_coefficients: torch.Tensor
    @property
    def feature(self):
        return torch.hstack((self.ndf.feature, self.bsdf_sh_coefficients.flatten()))


def evaluate_sggx(sggx: SGGX, ws: torch.Tensor) -> torch.Tensor:
    s = (1.0 / sggx.eigenvalues) * torch.square(ws @ sggx.eigenvectors.T)
    return 1.0 / torch.square(s.sum(-1))

def evaluate_disney_bsdf(metallic: torch.Tensor, roughness: torch.Tensor, ior: torch.Tensor, base_color: torch.Tensor, wi_local: torch.Tensor, wo_local: torch.Tensor) -> torch.Tensor:
    roughness2 = torch.square(roughness)
    eps = torch.finfo(torch.float32).eps
    def evaluate_disney_lambda(w: torch.Tensor) -> torch.Tensor:
        tan2theta = torch.square(w[..., 2])
        tan2theta = torch.where(tan2theta < eps, np.float32(0.0), (1.0 - tan2theta) / tan2theta)
        return (-1.0 + torch.sqrt(1.0 + roughness2 * tan2theta)) * 0.5
    def evaluate_disney_d(w: torch.Tensor) -> torch.Tensor:
        nominator = roughness2 * torch.where(w[..., 2] > 0.0, np.float32(1.0), np.float32(0.0))
        denominator = torch.pi * torch.square(1.0 + (roughness2 - 1.0) * torch.square(w[..., 2]))
        return nominator / (denominator + eps)
    wh = wi_local + wo_local
    wh[:] /= (torch.linalg.norm(wh, dim=-1) + torch.finfo(torch.float32).eps).unsqueeze(-1)
    fc = torch.pow(1.0 - (wi_local * wh).sum(-1), 5.0)
    reflectance = torch.square((ior - 1.0) / (ior + 1.0))
    f0 = metallic.unsqueeze(-1) * base_color + ((1.0 - metallic) * reflectance).unsqueeze(-1)
    f = f0 + fc.unsqueeze(-1) * (1.0 - f0)
    g = 1.0 / (1.0 + evaluate_disney_lambda(wi_local) + evaluate_disney_lambda(wo_local))
    d = evaluate_disney_d(wh)
    specular_bsdf = (d * g).unsqueeze(-1) * f
    specular_bsdf /= (4.0 * torch.abs(wi_local[..., 2]) * torch.abs(wo_local[..., 2]) + torch.finfo(torch.float32).eps).unsqueeze(-1)
    front_facing = torch.where(torch.minimum(wi_local[..., 2], wo_local[..., 2]) > 0.0, np.float32(1.0), np.float32(0.0))
    diffuse_bsdf = ((1.0 - fc) * (1.0 - metallic) * (1.0 - reflectance)).unsqueeze(-1) * base_color / torch.pi
    return (specular_bsdf + diffuse_bsdf) * front_facing.unsqueeze(-1)

def evaluate_aggregate(aggregate: Aggregate, wi: torch.Tensor, wo: torch.Tensor, num_samples: int = 32, batch_size: int = 1024) -> torch.Tensor:
    wi, wo = wi.to(torch.float32), wo.to(torch.float32)
    evaluation = torch.empty(wi.shape[:-1] + (3,), dtype=torch.float32, device=wi.device)
    for i in range(0, wi.shape[0], batch_size):
        batch_slice = slice(i, min(i + batch_size, wi.shape[0]))
        batch_wi, batch_wo = wi[batch_slice], wo[batch_slice]
        onb_wo = make_onb(batch_wo)
        ns = sample_hemisphere_cosine(batch_wi.shape[0] * num_samples, wi.device).reshape(batch_wi.shape[0], num_samples, 3)
        ns = ns @ onb_wo
        onb_ns = make_onb(ns)
        wi_local = torch.einsum('nmji,ni->nmj', onb_ns, batch_wi).reshape(-1, 3)
        wo_local = torch.einsum('nmji,ni->nmj', onb_ns, batch_wo).reshape(-1, 3)
        ndf_eval = evaluate_sggx(aggregate.ndf, ns.reshape(-1, 3))
        batch_sh_basis = evaluate_sh(ns.reshape(-1, 3))
        batch_bsdf_parameters = batch_sh_basis @ aggregate.bsdf_sh_coefficients
        metallic = batch_bsdf_parameters[..., 0].clamp_(0.0, 1.0)
        roughness = batch_bsdf_parameters[..., 1].clamp_(0.01, 1.0)
        ior = batch_bsdf_parameters[..., 2].clamp_(1.0, 2.0)
        base_color = batch_bsdf_parameters[..., 3:].clamp_(0.0, 1.0)
        batch_bsdf_eval = evaluate_disney_bsdf(metallic, roughness, ior, base_color, wi_local, wo_local)
        # batch_bsdf_eval = torch.ones(batch_bsdf_parameters.shape[:-1] + (3,), dtype=torch.float32, device=wi.device)
        nominator = ((ndf_eval * wi_local[..., 2].clamp_min(0.0)).unsqueeze(-1) * batch_bsdf_eval).reshape(-1, num_samples, 3).sum(-2)
        denominator = ndf_eval.reshape(-1, num_samples).sum(-1) + torch.finfo(torch.float32).eps
        evaluation[batch_slice] = nominator / denominator.unsqueeze(-1)
    return evaluation

def generate_random_aggregate(device: torch.device) -> Aggregate:
    ws, ys = evaluate_sf(4, device), torch.rand(4, 6, dtype=torch.float32, device=device)
    ws = ws @ sample_onb_uniform(1, device).squeeze(0)
    ys[..., 2] += 1.0
    bsdf_sh_coefficients = fit_sh_coefficients(ws, ys)
    eigenvectors = sample_onb_uniform(1, device).squeeze(0)
    eigenvalues = 1.0 - torch.rand(3, dtype=torch.float32, device=device)
    # eigenvalues = torch.ones(3, dtype=torch.float32, device=device)
    return Aggregate(SGGX(eigenvectors, eigenvalues), bsdf_sh_coefficients)

def generate_pretraining_dataset(num_aggregates: int, sphere_resolution: int, device: torch.device):
    dataset = torch.empty(num_aggregates, sphere_resolution * sphere_resolution, 39, dtype=torch.float32, device=device)
    for i in range(num_aggregates):
        aggregate = generate_random_aggregate(device)
        dataset[i, ..., :30] = aggregate.feature
        wi = evaluate_sf(sphere_resolution, device) @ sample_onb_uniform(1, device).squeeze(0)
        wo = evaluate_sf(sphere_resolution, device) @ sample_onb_uniform(1, device).squeeze(0)
        wi = torch.tile(wi, (sphere_resolution, 1)) # 1, 1, 1,...
        wo = wo.repeat(1, sphere_resolution).reshape(-1, 3)
        dataset[i, ..., 30:33] = wi
        dataset[i, ..., 33:36] = wo
        dataset[i, ..., 36:] = evaluate_aggregate(aggregate, wi, wo, 32)
    dataset[..., -3:] = torch.log(dataset[..., -3:] + 1.0)
    torch.save(dataset.reshape(-1, 39), "pretraining-dataset.pt")

def generate_verification_dataset(square_resolution: int, device: torch.device):
    aggregate = generate_random_aggregate(device)
    wi, wo = generate_visualization_ws(square_resolution)
    wi = torch.tensor(wi, dtype=torch.float32, device=device)
    wo = torch.tensor(wo, dtype=torch.float32, device=device)
    dataset = torch.empty(square_resolution ** 4, 36, dtype=torch.float32, device=device)
    dataset[..., :30] = aggregate.feature
    dataset[..., 30:33] = wi
    dataset[..., 33:36] = wo
    torch.save(dataset, "verification-dataset.pt")
    image = evaluate_aggregate(aggregate, wi, wo, 32).reshape(square_resolution ** 2, square_resolution ** 2, -1)
    save_exr_image(image, "reference.exr")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_pretraining_dataset(1024, 32, device)
    generate_verification_dataset(32, device)

if __name__ == "__main__":
    main()
