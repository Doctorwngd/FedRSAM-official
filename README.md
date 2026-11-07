# FedRSAM

## Setup

Use Linux, Python 3.8, and an NVIDIA GPU with CUDA support. Run the commands from the project root.

```bash
python -m pip install -r requirements.txt
mkdir -p data log plot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export FEDRSAM_TINY_CLASS_ORDER="$PWD/partitions/tiny_class_order.json"
unset FEDRSAM_RAY_ADDRESS RAY_ADDRESS

for split in partitions/num_workers_*; do
  if [ ! -e "${split##*/}" ]; then
    ln -s "$split" "${split##*/}"
  fi
done
```

The supplied client splits are in `partitions/`. The setup links them into the project root for the vision scripts.

CIFAR-10 and CIFAR-100 download automatically to `data/`.

For Tiny-ImageNet, extract the archive to `data/tiny-imagenet-200/` and save the ImageNet-21k ViT-B/16 checkpoint as `vit_base_patch16_224_in21k.pth` in the project root. [Tiny-ImageNet](https://cs231n.stanford.edu/tiny-imagenet-200.zip) · [ViT-B/16 checkpoint](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth)

For GLUE, put the training and validation JSONL files in `data/qnli/`, `data/MNLI/`, `data/RTE/`, and `data/QQP/`, and put the RoBERTa-Base files in `roberta_base/`. The commands below prepare those paths. [GLUE](https://huggingface.co/datasets/nyu-mll/glue) · [RoBERTa-Base](https://huggingface.co/FacebookAI/roberta-base)

```bash
python - <<'PY'
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download

for task, folder, first, second, validation in (
    ("qnli", "qnli", "question", "sentence", "validation"),
    ("mnli", "MNLI", "premise", "hypothesis", "validation_matched"),
    ("rte", "RTE", "sentence1", "sentence2", "validation"),
    ("qqp", "QQP", "question1", "question2", "validation"),
):
    target = Path("data") / folder
    target.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("nyu-mll/glue", task)
    for split, name in (("train", "train"), (validation, "validation")):
        rows = dataset[split].rename_columns({first: "text1", second: "text2"})
        rows.to_json(str(target / (name + ".jsonl")))

snapshot_download(
    "FacebookAI/roberta-base",
    local_dir="roberta_base",
    allow_patterns=["config.json", "model.safetensors", "vocab.json", "merges.txt", "tokenizer*.json"],
)
PY
```

## Vision training

Set `--gpu` to your CUDA device index. 
### CIFAR-10 / LeNet-5

```bash
python src/vision/train.py \
  --alg FedRSAM --data_name CIFAR10 --CNN lenet5 --epoch 301 --seed 3407 \
  --lr 1e-1 --E 5 --K 50 --batch_size 50 --normalization GN \
  --num_workers 100 --selection 0.1 --alpha_value 0.1 --gpu 0 --num_gpus_per 0.1 \
  --rho 0.1 --lambda_z 1e-3 --z_lr 1e-3 --alpha_lr 0.5 \
  --extname cifar10-lenet5 --output_dir runs/cifar10-lenet5
```

### CIFAR-10 / ResNet-18

```bash
python src/vision/train.py \
  --alg FedRSAM --data_name CIFAR10 --CNN resnet18 --epoch 301 --seed 3407 \
  --lr 1e-1 --E 5 --K 50 --batch_size 50 --normalization GN \
  --num_workers 100 --selection 0.1 --alpha_value 0.1 --gpu 0 --num_gpus_per 0.1 \
  --rho 0.1 --lambda_z 3e-4 --z_lr 1e-3 --alpha_lr 0.5 \
  --extname cifar10-resnet18 --output_dir runs/cifar10-resnet18
```

### CIFAR-100 / LeNet-5

```bash
python src/vision/train.py \
  --alg FedRSAM --data_name CIFAR100 --CNN lenet5 --epoch 301 --seed 3407 \
  --lr 1e-1 --E 5 --K 50 --batch_size 50 --normalization GN \
  --num_workers 100 --selection 0.1 --alpha_value 0.1 --gpu 0 --num_gpus_per 0.1 \
  --rho 0.1 --lambda_z 1e-3 --z_lr 1e-3 --alpha_lr 0.5 \
  --extname cifar100-lenet5 --output_dir runs/cifar100-lenet5
```

### CIFAR-100 / ResNet-18

```bash
python src/vision/train.py \
  --alg FedRSAM --data_name CIFAR100 --CNN resnet18 --epoch 301 --seed 3407 \
  --lr 1e-1 --E 5 --K 50 --batch_size 50 --normalization GN \
  --num_workers 100 --selection 0.1 --alpha_value 0.1 --gpu 0 --num_gpus_per 0.1 \
  --rho 0.1 --lambda_z 1e-3 --z_lr 1e-3 --alpha_lr 0.5 \
  --extname cifar100-resnet18 --output_dir runs/cifar100-resnet18
```

### Tiny-ImageNet / ViT-B/16

```bash
python src/tiny/runner.py \
  --alg FedRSAM --rounds 101 --seed 3407 --gpu 0 \
  --lambda-z 1e-3 --alpha-lr 0.5 --output runs/tiny-imagenet
```

## Language training

Set `--gpu` to your CUDA device index. The language runner saves logs to `log/` and curves to `plot/`. The `--mm` parameter controls normalized server momentum.

### QNLI / RoBERTa-Base

```bash
python src/language/new_llm_sam2.py \
  --alg FedRSAM --data_name qnli --CNN roberta_base --lr 1e-1 --epoch 101 \
  --E 10 --K 50 --batch_size 32 --num_workers 20 --selection 0.2 \
  --alpha_value 0.1 --gpu 0 --num_gpus_per 0.25 --mm 0.0 --rho 0.1 \
  --lambda_z 1.0 --z_lr 1e-3 --alpha_lr 0.5
```

### MNLI / RoBERTa-Base

```bash
python src/language/new_llm_sam2.py \
  --alg FedRSAM --data_name MNLI --CNN roberta_base --lr 1e-1 --epoch 101 \
  --E 10 --K 50 --batch_size 32 --num_workers 20 --selection 0.2 \
  --alpha_value 0.1 --gpu 0 --num_gpus_per 0.25 --mm 0.3 --rho 0.1 \
  --lambda_z 1.0 --z_lr 1e-3 --alpha_lr 0.5
```

### RTE / RoBERTa-Base

```bash
python src/language/new_llm_sam2.py \
  --alg FedRSAM --data_name RTE --CNN roberta_base --lr 1e-1 --epoch 101 \
  --E 10 --K 50 --batch_size 32 --num_workers 20 --selection 0.2 \
  --alpha_value 0.1 --gpu 0 --num_gpus_per 0.25 --mm 0.2 --rho 0.1 \
  --lambda_z 1.0 --z_lr 1e-2 --alpha_lr 0.5
```

### QQP / RoBERTa-Base

```bash
python src/language/new_llm_sam2.py \
  --alg FedRSAM --data_name QQP --CNN roberta_base --lr 1e-1 --epoch 101 \
  --E 10 --K 50 --batch_size 32 --num_workers 20 --selection 0.2 \
  --alpha_value 0.1 --gpu 0 --num_gpus_per 0.25 --mm 0.0 --rho 0.05 \
  --lambda_z 0.3 --z_lr 1e-3 --alpha_lr 0.5
```
