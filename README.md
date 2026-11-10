# Expandable Subspace Ensemble

A PyTorch implementation exploring expanding adapter subspaces for
class-incremental learning on CIFAR-10, based on the EASE paper.

The implementation in [ease.py](ease.py) uses a frozen pretrained ResNet-18,
a bottleneck adapter for each task, class prototypes, and prototype synthesis
across adapter subspaces. The demonstration partitions CIFAR-10 into five
two-class tasks and evaluates the classes seen so far.

This is a study implementation. It does not reproduce the original paper's
full benchmark protocol or establish its reported accuracy.

## Installation

Create a Python environment with [uv](https://docs.astral.sh/uv/), then install
the dependencies used by the implementation:

```sh
git clone https://github.com/stochi0/expandable-subspace-ensemble-cil.git
cd expandable-subspace-ensemble-cil
uv venv --python 3.11
uv pip install torch torchvision numpy
```

The pretrained-backbone path uses torchvision's `ResNet18_Weights` API.
There is no locked dependency set in this repository.

## A small adapter example

```python
import torch
from ease import Adapter

adapter = Adapter(d=512, r=16)
features = torch.randn(4, 512)
adapted = adapter(features)
print(adapted.shape)  # torch.Size([4, 512])
```

The example exercises the adapter without downloading data or pretrained weights.
Run it with the environment's Python, for example `uv run --no-project python`.

## Full demonstration and limitations

The entrypoint is `ease.py`; there are no `main.py` or `app.py` entrypoints.
The full demonstration downloads CIFAR-10 and pretrained ResNet-18 weights,
then trains each new adapter for ten epochs.

The cumulative evaluation loop currently assembles DataLoader batches with
`torch.stack` and constructs labels from a list of batch tensors. That assembly
needs correction before the full demonstration can supply valid end-to-end
accuracy measurements. Use the adapter example for a local smoke check; no
benchmark accuracy is claimed here.

Device selection is available through `run(device="cpu")`, `run(device="mps")`,
or `run(device="cuda")`. The script's default `auto` path selects MPS when
available and otherwise CPU; request CUDA explicitly for an available CUDA device.

## References

- [Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning](docs/cvpr2024_Expandable_Subspace_Ensemble_CIL.pdf) — the original paper.
- [Concise review](docs/concise_review.pdf) — accompanying study notes.

## Citation

<!-- BEGIN generated citations -->

```bibtex
@software{bodadeprojectexpandablesubspaceensemblecil,
  author = {Bodade, Ayush},
  title = {{Expandable Subspace Ensemble}},
  url = {https://github.com/stochi0/expandable-subspace-ensemble-cil},
  note = {Software implementation; cite the original research separately. Specify the commit or version used.}
}
```

<!-- END generated citations -->
