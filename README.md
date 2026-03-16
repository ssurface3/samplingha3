# Optional Homework: Diffusion Samplers

## Installation
- python 3.10.14
- jax 0.6.2

We recommend using the conda (or mamba) environment to install the dependencies.
```bash
conda create -n gfn-smc-jax python=3.10.14
conda activate gfn-smc-jax
```

Install tensorflow first since it sometimes causes conflicts with other packages.
```bash
pip install tensorflow==2.16.1
```

Install the jax and jaxlib with the appropriate CUDA version or TPU support, e.g., cuda12
```bash
pip install -U "jax[cuda12]==0.6.2"
```

Install the other dependencies.
```bash
pip install -r requirements.txt
```

## Usage

Here we mainly focus on the GFlowNet-based algorithms. 

Basic usage:
```bash
python run.py algorithm=<algorithm_name> target=<target_name>
```

`<algorithm_name>` can be one of the following:
- `gfn_tb` baseline
- `gfn_tb_learn_bwd` what you need to implement

`<target_name>` can be one of the following:
- `gaussian_mixture`
