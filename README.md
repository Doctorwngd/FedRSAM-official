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

## Run commands

Set `--gpu` to the CUDA device index. Vision outputs go to `runs/`. The language runner writes text logs to `log/`, curves to `plot/`, and TensorBoard data to `runs/`.

### CIFAR-10 / LeNet-5

```bash
python src/vision/train.py \
  --alg FedRSAM --protocol original_draw3 --data_name CIFAR10 --model lenet5 \
  --normalization GN --preprocessing_profile fedrsam_legacy --rounds 301 --num_workers 100 \
  --selection 0.1 --local_epochs 5 --max_steps 50 --batch_size 50 --lr 0.1 --lr_decay 0.998 \
  --lr_schedule cosine --weight_decay 0.001 --rho 0.1 --rho_schedule constant \
  --rho_schedule_rounds 0 --inner_steps 1 --alpha_lr 0.5 --alpha_scope global \
  --cone_scope global --z_lr 0.001 --lambda_z 0.001 --fixed_axis_until 0 \
  --regularizer_solver closed_form --restricted_impl fast --search_diagnostics minimal \
  --alpha_init 0.1 --server_momentum 0.85 --client_lookahead_gamma 0.0 --warmup_rho 0.05 \
  --project_z_box --dirichlet_alpha 0.1 --num_gpus_per_client 0.1 --seed 3407 \
  --eval_every 10 --eval_batch_size 200 --num_eval_workers 4 --deterministic_test \
  --geometry_every 1 --geometry_projection_steps 5 --geometry_rank_samples 4096 \
  --history_profile off --alpha 10.0 --p 1.0 --pre 1 --print 0 --pix 32.0 --lora 0 \
  --save_best --gpu 0 --data_root data --output_dir runs/cifar10-lenet5 \
  --run_name cifar10-lenet5
```

### CIFAR-10 / ResNet-18

```bash
python src/vision/train.py \
  --alg FedRSAM --protocol original_draw3 --data_name CIFAR10 --model resnet18 \
  --normalization GN --preprocessing_profile fedrsam_legacy --rounds 301 --num_workers 100 \
  --selection 0.1 --local_epochs 5 --max_steps 50 --batch_size 50 --lr 0.1 --lr_decay 0.998 \
  --lr_schedule cosine --weight_decay 0.001 --rho 0.1 --rho_schedule constant \
  --rho_schedule_rounds 0 --inner_steps 1 --alpha_lr 0.5 --alpha_scope global \
  --cone_scope global --z_lr 0.001 --lambda_z 0.0003 --fixed_axis_until 0 \
  --regularizer_solver closed_form --restricted_impl fast --search_diagnostics minimal \
  --alpha_init 0.1 --server_momentum 0.85 --client_lookahead_gamma 0.0 --warmup_rho 0.05 \
  --project_z_box --dirichlet_alpha 0.1 --num_gpus_per_client 0.1 --seed 3407 \
  --eval_every 10 --eval_batch_size 200 --num_eval_workers 4 --deterministic_test \
  --geometry_every 1 --geometry_projection_steps 5 --geometry_rank_samples 4096 \
  --history_profile off --alpha 10.0 --p 1.0 --pre 1 --print 0 --pix 32.0 --lora 0 \
  --save_best --gpu 0 --data_root data --output_dir runs/cifar10-resnet18 \
  --run_name cifar10-resnet18
```

### CIFAR-100 / LeNet-5

```bash
python src/vision/train.py \
  --alg FedRSAM --protocol original_draw3 --data_name CIFAR100 --model lenet5 \
  --normalization GN --preprocessing_profile fedrsam_legacy --rounds 301 --num_workers 100 \
  --selection 0.1 --local_epochs 5 --max_steps 50 --batch_size 50 --lr 0.1 --lr_decay 0.998 \
  --lr_schedule cosine --weight_decay 0.001 --rho 0.1 --rho_schedule constant \
  --rho_schedule_rounds 0 --inner_steps 1 --alpha_lr 0.5 --alpha_scope global \
  --cone_scope global --z_lr 0.001 --lambda_z 0.001 --fixed_axis_until 0 \
  --regularizer_solver closed_form --restricted_impl fast --search_diagnostics minimal \
  --alpha_init 0.1 --server_momentum 0.85 --client_lookahead_gamma 0.0 --warmup_rho 0.05 \
  --project_z_box --dirichlet_alpha 0.1 --num_gpus_per_client 0.1 --seed 3407 \
  --eval_every 10 --eval_batch_size 200 --num_eval_workers 4 --deterministic_test \
  --geometry_every 1 --geometry_projection_steps 5 --geometry_rank_samples 4096 \
  --history_profile off --alpha 10.0 --p 1.0 --pre 1 --print 0 --pix 32.0 --lora 0 \
  --save_best --gpu 0 --data_root data --output_dir runs/cifar100-lenet5 \
  --run_name cifar100-lenet5
```

### CIFAR-100 / ResNet-18

```bash
python src/vision/train.py \
  --alg FedRSAM --protocol original_draw3 --data_name CIFAR100 --model resnet18 \
  --normalization GN --preprocessing_profile fedrsam_legacy --rounds 301 --num_workers 100 \
  --selection 0.1 --local_epochs 5 --max_steps 50 --batch_size 50 --lr 0.1 --lr_decay 0.998 \
  --lr_schedule cosine --weight_decay 0.001 --rho 0.1 --rho_schedule constant \
  --rho_schedule_rounds 0 --inner_steps 1 --alpha_lr 0.5 --alpha_scope global \
  --cone_scope global --z_lr 0.001 --lambda_z 0.001 --fixed_axis_until 0 \
  --regularizer_solver closed_form --restricted_impl fast --search_diagnostics minimal \
  --alpha_init 0.1 --server_momentum 0.85 --client_lookahead_gamma 0.0 --warmup_rho 0.05 \
  --project_z_box --dirichlet_alpha 0.1 --num_gpus_per_client 0.1 --seed 3407 \
  --eval_every 10 --eval_batch_size 200 --num_eval_workers 4 --deterministic_test \
  --geometry_every 1 --geometry_projection_steps 5 --geometry_rank_samples 4096 \
  --history_profile off --alpha 10.0 --p 1.0 --pre 1 --print 0 --pix 32.0 --lora 0 \
  --save_best --gpu 0 --data_root data --output_dir runs/cifar100-resnet18 \
  --run_name cifar100-resnet18
```

### Tiny-ImageNet / ViT-B/16

```bash
python src/tiny/runner.py \
  --alg FedRSAM_CF --rounds 101 --seed 3407 --lambda-z 0.001 --alpha-lr 0.5 --gpu 0 --root . \
  --output runs/tiny-imagenet
```

### QNLI / RoBERTa-Base

```bash
python src/language/new_llm_sam2.py \
  --alg SAMFed --lr 1e-1 --data_name qnli --CNN roberta_base --optimizer SGD \
  --alpha_value 0.1 --alpha 0.9 --epoch 101 --lr_decay 2 --gamma 0.0 \
  --E 10 --batch_size 32 --K 50 --num_workers 20 --selection 0.2 --gpu 0 \
  --p 1 --num_gpus_per 0.25 --pre 1 --preprint 5 --freeze 1 --r 16 --lora 1 --print 1 \
  --rho 0.1 --lambda_z 1e0 --z_lr 1e-3 --alpha_lr 0.5 --extname RoBERTa_SST2
```

### MNLI / RoBERTa-Base

```bash
python src/language/new_llm_sam2.py \
  --alg SAMFed --lr 1e-1 --data_name MNLI --CNN roberta_base --optimizer SGD \
  --alpha_value 0.1 --alpha 0.9 --epoch 101 --lr_decay 2 --gamma 0.3 \
  --E 10 --batch_size 32 --K 50 --num_workers 20 --selection 0.2 --gpu 0 \
  --p 1 --num_gpus_per 0.25 --pre 1 --preprint 5 --freeze 1 --r 16 --lora 1 --print 1 \
  --rho 0.1 --lambda_z 1e0 --z_lr 1e-3 --alpha_lr 0.5 --extname RoBERTa_SST2
```

### RTE / RoBERTa-Base

```bash
python src/language/new_llm_sam2.py \
  --alg SAMFed --lr 1e-1 --data_name RTE --CNN roberta_base --optimizer SGD \
  --alpha_value 0.1 --alpha 0.9 --epoch 101 --lr_decay 2 --gamma 0.2 \
  --E 10 --batch_size 32 --K 50 --num_workers 20 --selection 0.2 --gpu 0 \
  --p 1 --num_gpus_per 0.25 --pre 1 --preprint 5 --freeze 1 --r 16 --lora 1 --print 1 \
  --rho 0.1 --lambda_z 1e0 --z_lr 1e-2 --alpha_lr 0.5 --extname RoBERTa_SST2
```

### QQP / RoBERTa-Base

```bash
python src/language/new_llm_sam2.py \
  --alg SAMFed --lr 1e-1 --data_name QQP --CNN roberta_base --optimizer SGD \
  --alpha_value 0.1 --alpha 0.9 --epoch 101 --lr_decay 2 --gamma 0.0 \
  --E 10 --batch_size 32 --K 50 --num_workers 20 --selection 0.2 --gpu 0 \
  --p 1 --num_gpus_per 0.25 --pre 1 --preprint 5 --freeze 1 --r 16 --lora 1 --print 1 \
  --rho 0.05 --lambda_z 0.3 --z_lr 1e-3 --alpha_lr 0.5 --extname RoBERTa_SST2
```
