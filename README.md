# novaCTF
NovaCTF is a freeware for 3D-CTF correction for electron microscopy, as described in this publication:

_Turoňová, B., Schur, F.K.M, Wan, W. and Briggs, J.A.G. Efficient 3D-CTF correction for cryo-electron tomography using NovaCTF improves subtomogram averaging resolution to 3.4 Å._  
_([doi.org/10.1016/j.jsb.2017.07.007](https://doi.org/10.1016/j.jsb.2017.07.007))_


The source code for NovaCTF is distributed under an GPL v.3 license.  

For more information how to install and use NovaCTF see Wiki pages.  

In case of any problems or if you have feedback for use do not hesitate to write us (or open an issue here).

## GPU Acceleration

This repository now keeps the original FFTW-based CPU implementation as the default execution path and adds an optional NVIDIA CUDA backend for the FFT-heavy hotspots in the reconstruction pipeline:

- `FFTRoutines::real2DTransform`
- `FFTRoutines::complex2DTransform`
- `FFTRoutines::many1DTransform`
- `FFTRoutines::real1DTransform`

The upper-level tomography, CTF correction and filtering logic is unchanged. Only the low-level FFT execution layer is switched when CUDA support is enabled at build time.

### Dependencies

- CPU build: FFTW3 and FFTW3f
- GPU build: FFTW3, FFTW3f, CUDA Toolkit with cuFFT and a supported NVIDIA GPU

On cluster environments, make sure the compiler, FFTW and CUDA modules are loaded before building.

### Build

CPU-only build remains unchanged:

```bash
module load gnu9/9.4.0
module load fftw/3.3.8
make
```

Enable the CUDA backend explicitly:

```bash
module load gnu9/9.4.0
module load cuda/11.8
module load fftw/3.3.8
make USE_CUDA=1 CUDA_HOME=/apps/cuda-11.8
```

If CUDA is installed outside the default path, provide `CUDA_HOME` or `CUDA_PATH`:

```bash
make USE_CUDA=1 CUDA_HOME=/usr/local/cuda
```

On Windows with GNU Make, `CUDA_PATH` is typically provided by the CUDA Toolkit installer.

If you only want to check the CUDA branch compilation without linking the whole program, build the two critical objects directly:

```bash
nvcc -O3 -std=c++14 -DUSE_CUDA -Isrc -I${CUDA_HOME}/include -c src/cudaFftRoutines.cu -o src/cudaFftRoutines.o
g++ -O3 -DNDEBUG -std=c++14 -DUSE_CUDA -I${CUDA_HOME}/include -c src/fftRoutines.cpp -o src/fftRoutines.o
```

### Runtime Behavior And Fallback

- Without `USE_CUDA=1`, the binary is identical in behavior to the original CPU-only build.
- With `USE_CUDA=1`, the program attempts to use the CUDA FFT backend for the supported FFT routines.
- If CUDA initialization fails or no device is available, the code falls back to the original FFTW implementation automatically.
- Set `NOVACTF_DISABLE_CUDA=1` to force the CPU path even in a CUDA-enabled build.
- On GPU clusters, login nodes may not expose CUDA devices. Run GPU validation and production jobs on a compute node with visible GPUs.

### Validation

A minimal backend comparison executable is provided for CUDA-enabled builds:

```bash
module load gnu9/9.4.0
module load cuda/11.8
module load fftw/3.3.8
make USE_CUDA=1 fft_backend_validation CUDA_HOME=/apps/cuda-11.8
./fft_backend_validation
```

The validation program runs the same FFT entry points once through the CPU path and once through the GPU path, then compares the outputs.

Small numerical differences are expected because FFTW and cuFFT do not guarantee bitwise-identical floating-point accumulation order. The validation therefore checks for bounded numerical error instead of exact equality.

A successful run looks like:

```text
real2DTransform max abs diff: 0
complex2DTransform max abs diff: 0
many1DTransform roundtrip max abs diff: 5.96046e-07
FFT backend validation passed.
```

If the program reports `CUDA runtime unavailable: no CUDA-capable device is detected`, rebuild is not required; rerun the validation on a GPU node.

If the program reports a missing `libcufft.so`, rebuild with an explicit `CUDA_HOME` that matches the loaded CUDA module so the correct runtime library path is embedded.
