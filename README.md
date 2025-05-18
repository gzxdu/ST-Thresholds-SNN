```markdown
# ST-Thresholds-SNN

This repository implements **Adaptation and learning of spatio-temporal thresholds in spiking neural networks** in Spiking Neural Networks (SNNs), supporting training on both frame-based and event-based vision datasets.


## Requirements

- Python >= 3.9
- PyTorch >= 1.13

Install dependencies:
```bash
pip install torch>=1.13
```

## Usage

### 1. Training on CIFAR10

To train a model on CIFAR10, use:

```bash
python main_train.py --dataset CIFAR10
```

You can also select other datasets (if supported) using the `--dataset` argument.

### 2. Training on CIFAR10-DVS

To train on the event-based CIFAR10-DVS dataset, use:

```bash
python main_train_dvs_TET.py
```

## File Structure

- `main_train.py`: Training script for frame-based datasets.
- `main_train_dvs_TET.py`: Training script for event-based datasets.
- Other modules: Model definitions, utilities, and configuration files.

## Reference

If you find this repository helpful, please cite:

```bibtex
@article{fu2025adaptation,
  title={Adaptation and learning of spatio-temporal thresholds in spiking neural networks},
  author={Fu, Jiahui and Gou, Shuiping and Wang, Peizhao and Jiao, Licheng and Guo, Zhang and Li, Jisheng and Liu, Rong},
  journal={Neurocomputing},
  pages={130423},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgments

- This project is based on [MPBN](https://github.com/yfguo91/MPBN) codebase.
- Uses modules and inspiration from the [SpikingJelly](https://github.com/fangwei123456/spikingjelly) library.