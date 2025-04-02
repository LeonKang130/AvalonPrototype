# AvalonPrototype

## Dataset Generation

### Pretraining Dataset

The encoder and decoder are pretrained using artificial aggregated appearance data, which are basically just microflakes associated with an SGGX normal distribution and a simplified Disney BSDF. We feed the parameters of these microflake models to the encoder, and feed the latent code along with $\omega_i,\omega_o$ to the decoder to predict aggregated RGB color. The labels are generated through numerical integration of the input microflake models.

Since we have:

$$
\hat f_\mathrm{novis}=\frac{1}{|A|_{\omega_o}}\int_Af(x,\omega_i,\omega_o)\langle n_x,\omega_i\rangle\langle n_x,\omega_o\rangle\mathrm{d}x
$$

We assume that the base Disney BSDF is described by orientation-varying parameters $\beta(n)$, and the normal distribution of the microflakes (or surfels on $A$) is given statistically by $D(n)$, then we can write the integral as:

$$
\hat f_\mathrm{novis}=\frac{1}{|A|_{\omega_o}}\int_{S^2}f(\omega_i,\omega_o;\beta(n))\langle n,\omega_i\rangle\langle n,\omega_o\rangle D(n)\mathrm{d}n
$$

Since the value of $|A|_\mathrm{\omega_o}$ itself is also given by an integral on $A$, we can rewrite it as an integral on $S^2$ as well:

$$
|A|_{\omega_o}=\int_A\langle n_x,\omega_p\rangle\mathrm{d}x=\int_{S^2}\langle n,\omega_o\rangle D(n)\mathrm{d}n
$$

Taking $N$ samples of $n$ from the sphere, we can approximate $\hat f_\mathrm{novis}$ as:

$$
\hat f_\mathrm{novis}\approx\frac{\sum_{k=1}^Nf(\omega_i,\omega_o;\beta(n_k))\langle n_k,\omega_i\rangle\langle n_k,\omega_o\rangle D(n_k)/p(n_k)}{\sum_{k=1}^N\langle n_k,\omega_o\rangle D(n_k)/p(n_k)}
$$

where $p(n_k)$ is the probability density function for normal sample $n_k$ to be selected. Currently, we use uniform sampling for $n_k$, but we will look into importance sampling in the future.

### Simulating Voxel Aggregation

To evaluate the behaviors of aggregated appearances from real triangles within a voxel, we generate random triangles within a voxel and numerically estimate the aggregated appearance by sampling points on the triangles. After that, we can use these surfels to evaluate the value of the aggregated appearance given different pairs of $\omega_i,\omega_o$ using the following equation:

$$
\hat f_\mathrm{novis}\approx
$$

