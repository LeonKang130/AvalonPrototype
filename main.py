from avalon import Avalon
from util import evaluate_octahedron
import numpy as np
import OpenEXR, Imath

def generate_visualization_uv(resolution: int) -> np.ndarray:
    i = np.arange(resolution * resolution)
    x = np.tile(i, resolution * resolution)[..., None]
    y = np.repeat(i, resolution * resolution)[..., None]
    xy = np.hstack((x, y))
    uv_wi = (xy % resolution + 0.5) / resolution
    uv_wo = (xy // resolution + 0.5) / resolution
    return np.hstack((uv_wi, uv_wo))

if __name__ == "__main__":
    # np.random.seed(5)
    avalon = Avalon()
    sphere_resolution = 20
    uv = generate_visualization_uv(sphere_resolution)
    wi = evaluate_octahedron(uv[..., :2])
    wo = evaluate_octahedron(uv[..., 2:])
    avalon_eval = (avalon.evaluate(wi, wo, 64)
        .reshape(sphere_resolution * sphere_resolution,
                 sphere_resolution * sphere_resolution, 3))
    exr_data = [avalon_eval[..., i].astype(np.float32).tobytes() for i in range(avalon_eval.shape[-1])]
    header = OpenEXR.Header(*avalon_eval.shape[:2])
    channels = ["R", "G", "B", "A"][:len(exr_data)]
    header["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in channels}
    exr_file = OpenEXR.OutputFile("avalon.exr", header)
    exr_file.writePixels(dict(zip(channels, exr_data)))
    exr_file.close()