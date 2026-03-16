# Optional Homework: Diffusion Samplers

## What you need to implement
- Model itself in `algorithms/common/models/pisgrad_net_learn_bwd.py`
- Model initialization in `algorithms/common/diffusion_related/init_model.py`
- Core logic of the algorithm with learnable backward in `algorithms/gfn_tb_learn_bwd/gfn_tb_rnd.py`

## Installation
- python 3.10.14
- jax 0.6.2

We recommend using the conda (or mamba) environment to install the dependencies.
```bash
conda create -n gfn-smc-jax python=3.10.14
conda activate gfn-smc-jax
```

Install dependencies.
```bash
pip install -r requirements.txt
```

## Usage

Here we mainly focus on the GFlowNet-based algorithms. 

Basic usage:
```bash
python run.py algorithm=<algorithm_name> target=<target_name>
```

To enable/disable WandB add `use_wandb=True` or `use_wandb=False`

`<algorithm_name>` can be one of the following:
- `gfn_tb` baseline
- `gfn_tb_learn_bwd` what you need to implement

`<target_name>` can be one of the following:
- `gaussian_mixture`

## Configs
You can check configuration of the algorithms in `configs/algorithm/gfn_tb.yaml` for the baseline and `configs/algorithm/gfn_tb_learn_bwd.yaml` for your implementation.