import math
import os

from sam import SAM
from samFedllm import SAMFed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=16, type=int, help='# batch_size')
parser.add_argument('--E', default=1, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedMoment', type=str, help='alg')
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0,1', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.99', type=float, help='lr_decay')
parser.add_argument('--data_name', default='imagenet', type=str, help='lr_decay')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')
parser.add_argument('--lr_ps', default='0.15', type=float, help='only for FedAdam ')
parser.add_argument('--alpha_value', default='0.6', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.06', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=1, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='VIT-L', type=str, help=' for mom_step')
parser.add_argument('--gamma', default=0.9, type=float, help=' for mom_step')
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
parser.add_argument('--p', default=1, type=int, help=' for mom_step')
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--datapath', type=str,
                    default="./data")
parser.add_argument('--num_gpus_per', default=0.5, type=float, help=' for mom_step')
parser.add_argument('--rho', default=0.1, type=float, help='rho')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD,AdamW')
parser.add_argument("--preprint", type=int, default=5, help="")
parser.add_argument("--R", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--lora", type=int, default=0, help="")
parser.add_argument("--AdaLora", type=int, default=0, help="")
parser.add_argument("--r", type=int, default=16, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--beta1", type=float, default=0.9, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--beta2", type=float, default=0.999, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--K', default=20, type=int, help='#workers')
parser.add_argument('--freeze', default=1, type=int, help='# batch_size')
parser.add_argument("--pre", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--print', default=0, type=int, help=' for mom_step')

parser.add_argument('--lambda_z', default=0.3, type=float, help='reg for z')
parser.add_argument('--z_lr', default=1e-3, type=float, help='learning rate for z')
parser.add_argument('--alpha_lr', default=0.5, type=float, help='learning rate for alpha')

parser.set_defaults(alpha=0.9, lr_decay=2.0, lora=1, extname='RoBERTa_SST2', print=1)
args = parser.parse_args()
print(args.lora)
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx


from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from torch.func import functional_call
import numpy as np
from torch.utils.data import SubsetRandomSampler, random_split
import random
from math import exp
from copy import deepcopy
import ray
from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from dirichlet_data import data_from_dirichlet

from peft import AdaLoraConfig, get_peft_model
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Adafactor, Trainer, TrainingArguments

from datasets import load_dataset, tqdm
from peft import LoraConfig, get_peft_model, TaskType


print(torch.cuda.is_available())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
num_gpus_per = args.num_gpus_per
num_gpus = len(gpu_idx.split(','))

data_name = args.data_name
CNN = args.CNN


if args.CNN=='roberta_base':
    model_path='./roberta_base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.r*2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=['query', 'value', 'key', 'dense'],
    )

    if args.AdaLora == 1:
        lora_config = AdaLoraConfig(
            target_r=args.r,
            init_r=args.r * 2,
            beta1=0.85,
            beta2=0.85,
            tinit=0,
            tfinal=500,
            deltaT=10,
            lora_alpha=args.r * 2,
            lora_dropout=0.05,
            target_modules=['query', 'value', 'key'],
            task_type=TaskType.SEQ_CLS,
        )

if args.data_name=='QQP':
    dataset_path = './data/QQP'
    dataset = load_dataset(dataset_path)
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]



if args.data_name=='MNLI':
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)
    dataset_path = './data/MNLI'
    dataset = load_dataset(dataset_path)
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]


if args.data_name=='STS-B':
    dataset_path = './data/sts-b'
    dataset = load_dataset(dataset_path)

    dataset = dataset.rename_column("score", "label")
    def preprocess_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length",
                         max_length=128)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)


    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name=='WNLI':
    dataset_path = './data/WNLI'
    dataset = load_dataset(dataset_path)
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]




if args.data_name=='RTE':
    dataset_path = './data/RTE'
    dataset = load_dataset(dataset_path)
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

if args.data_name=='MRPC':
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",max_length=128)
    dataset_path = './data/MRPC'
    dataset = load_dataset(dataset_path)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

if args.data_name=='qnli':
    def preprocess_function(examples):
        inputs = tokenizer(
            examples["text1"],
            examples["text2"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        labels = [1 if label == "entailment" else 0 for label in examples["label"]]
        inputs["labels"] = labels
        return inputs
    dataset_path = './data/qnli'
    dataset = load_dataset(dataset_path)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]


if args.data_name=='sst2':
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", return_tensors="pt",max_length=64)
    dataset_path = './data/sst2'
    dataset = load_dataset(dataset_path)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name=='cola':
    dataset_path = './data/cola'
    def preprocess_function(examples):
        return tokenizer(examples["Sentence"], padding="max_length", truncation=True,max_length=64)

    dataset = load_dataset(dataset_path)
    dataset = dataset.rename_column("Acceptability", "label")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]


seed = 42
if args.alpha_value==1:
    def get_data_loader(pid, data_idx, batch_size, data_name):
        generator = torch.Generator().manual_seed(42)
        train_dataset = tokenized_dataset["train"]
        total_size = len(train_dataset)
        subset_size = total_size // args.num_workers
        remainder = total_size % args.num_workers

        split_sizes = [subset_size] * (args.num_workers - 1) + [subset_size + remainder]
        subsets = random_split(train_dataset, split_sizes, generator=generator)
        sample_chosed = data_idx[pid]
        train_loader = DataLoader(subsets[pid], batch_size=args.batch_size, shuffle=True)
        return train_loader

if args.alpha_value!=1:
    def get_data_loader(pid, data_idx, batch_size, data_name):
        sample_chosed = data_idx[pid]
        train_sampler = SubsetRandomSampler(sample_chosed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler, num_workers=0, generator=torch.Generator().manual_seed(seed))
        return train_loader

def get_data_loader_test(data_name):
    test_dataset = tokenized_dataset["validation"]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4)
    return test_loader

def get_data_loader_train(data_name):
    train_dataset = tokenized_dataset["train"].select(range(1000))
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4)
    return test_loader

def evaluate2(model, test_loader, train_loader):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    start_time1 = time.time()
    print('evaluate')
    with torch.no_grad():
        for batch in tqdm(test_loader,disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss+= criterion(logits, target)
    accuracy = 100. * correct / total
    end_time1 = time.time()
    print('evaluate完毕', '    ', end_time1 - start_time1)
    return  accuracy , test_loss / len(test_loader), torch.tensor(0)

def evaluate(model, test_loader, train_loader):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    start_time1 = time.time()
    print('evaluate')
    with torch.no_grad():
        for batch in tqdm(test_loader,disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss+= criterion(logits, target)
        for batch in tqdm(train_loader, disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            train_loss += criterion(logits, target)
    accuracy = 100. * correct / total
    end_time1 = time.time()
    print('evaluate完毕', '    ', end_time1 - start_time1)
    return  accuracy , test_loss / len(test_loader), train_loss / len(train_loader)


@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if args.CNN == 'roberta_base':
            model_path = './roberta_base'
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
            if args.data_name == 'MNLI':
                self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)

        if args.lora == 1 and args.alg!="FLORA":
            self.model = get_peft_model(self.model, lora_config)
            print(args.lora)
        self.pid = pid
        self.num_workers = num_workers
        self.data_iterator = None
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
        self.lr_decay = lr_decay
        self.alg = alg
        self.data_idx = data_idx
        self.pre_ps_weight = None
        self.pre_loc_weight = None
        self.flag = False
        self.ci = {}
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.momen_v = {}
        self.momen_m = {}
        self.R =1
        self.t ={k:  torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in self.model.named_parameters()}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01,
                                           betas=(args.beta1, args.beta2), amsgrad=False)
        self.constraint_space = None
        self.constraint_space_gpu = None
        self.inner_steps = getattr(args, "inner_steps", 1)
        self.alpha_lr = getattr(args, "alpha_lr", 0.5)
        self.z_lr = getattr(args, "z_lr", 1e-3)
        self.lambda_z = getattr(args, "lambda_z", 1.5)


    def data_id_loader(self, index):
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)



    def state_id_loader(self, index, shared_state):
        self.ci = ray.get(shared_state.get_ci_dict.remote(index))




    def get_train_loss(self):
        return self.loss

    def get_param_name(self, param):
        for name, p in self.model.named_parameters():
            if p is param:
                return name
        return None

    def solve_restricted_delta(self, input_ids, attention_mask, target):
        named_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        param_names = [name for name, _ in named_params]
        D = self.constraint_space_gpu["D"]
        var_list = self.constraint_space_gpu["var"]
        inner_steps = self.inner_steps
        alpha_lr = self.alpha_lr
        z_lr = self.z_lr
        lambda_z = self.lambda_z

        main_dirs = []
        layer_vars = []
        for d_list, var in zip(D, var_list):
            main_dirs.append(d_list[0])
            layer_vars.append(var)

        alpha_raw_list = []
        z_list = []
        for _, param in named_params:
            alpha_raw = torch.tensor([0.1], device=param.device, dtype=param.dtype, requires_grad=True)
            z = torch.zeros_like(param, device=param.device, requires_grad=True)
            alpha_raw_list.append(alpha_raw)
            z_list.append(z)

        for _ in range(inner_steps):
            delta_list = []
            layer_regs = []
            for main_dir, var, alpha_raw, z in zip(main_dirs, layer_vars, alpha_raw_list, z_list):
                std = torch.sqrt(var + 1e-12)
                with torch.no_grad():
                    z.clamp_(-std, std)

                main_flat = main_dir.reshape(-1)
                z_flat = z.reshape(-1)
                z_perp_flat = z_flat - torch.dot(z_flat, main_flat) * main_flat
                z_perp = z_perp_flat.view_as(main_dir)

                alpha = alpha_raw * alpha_raw
                delta = alpha * main_dir + z_perp

                delta_list.append(delta)
                layer_regs.append(((z * z) / (var + 1e-12)).mean())

            reg_z = torch.stack(layer_regs).mean()

            param_and_buffer_dict = {}
            delta_map = {name: delta for name, delta in zip(param_names, delta_list)}
            for name, param in self.model.named_parameters():
                if name in delta_map:
                    param_and_buffer_dict[name] = param.detach() + delta_map[name]
                else:
                    param_and_buffer_dict[name] = param
            for name, buf in self.model.named_buffers():
                param_and_buffer_dict[name] = buf

            output = functional_call(
                self.model,
                param_and_buffer_dict,
                (),
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            logits = output.logits
            inner_obj = self.criterion(logits, target.long()) - lambda_z * reg_z
            grads = torch.autograd.grad(inner_obj, alpha_raw_list + z_list, create_graph=False, allow_unused=False)

            alpha_grads = grads[:len(alpha_raw_list)]
            z_grads = grads[len(alpha_raw_list):]
            with torch.no_grad():
                for alpha_raw, g in zip(alpha_raw_list, alpha_grads):
                    alpha_raw.add_(alpha_lr * g)
                for z, g, var in zip(z_list, z_grads, layer_vars):
                    z.add_(z_lr * g)
                    std = torch.sqrt(var + 1e-12)
                    z.clamp_(-std, std)

            alpha_raw_list = [alpha_raw.detach().requires_grad_(True) for alpha_raw in alpha_raw_list]
            z_list = [z.detach().requires_grad_(True) for z in z_list]

        final_delta_list = []
        final_layer_regs = []
        for main_dir, var, alpha_raw, z in zip(main_dirs, layer_vars, alpha_raw_list, z_list):
            std = torch.sqrt(var + 1e-12)
            with torch.no_grad():
                z.clamp_(-std, std)

            main_flat = main_dir.reshape(-1)
            z_flat = z.reshape(-1)
            z_perp_flat = z_flat - torch.dot(z_flat, main_flat) * main_flat
            z_perp = z_perp_flat.view_as(main_dir)
            delta = (alpha_raw * alpha_raw) * main_dir + z_perp

            final_delta_list.append(delta)
            final_layer_regs.append(((z * z) / (var + 1e-12)).mean())

        reg_z = torch.stack(final_layer_regs).mean()
        perturb_norm = torch.norm(torch.stack([delta.norm(p=2) for delta in final_delta_list]), p=2)
        scale = args.rho / (perturb_norm + 1e-12)
        final_delta_list = [delta * scale for delta in final_delta_list]

        debug_info = {
            "reg_z": reg_z.detach(),
            "perturb_norm": perturb_norm.detach(),
            "scale": scale.detach() if torch.is_tensor(scale) else torch.tensor(scale),
            "alpha_vals": [alpha_raw.detach().cpu() for alpha_raw in alpha_raw_list],
        }
        return final_delta_list, debug_info


    def update_fedavg(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-3)
        step = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_SAM(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)

        if args.freeze == 0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True

        base_optimizer = torch.optim.SGD
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = SAM(trainable_params, base_optimizer, lr=lr, weight_decay=0.001, rho=args.rho)

        step = 0
        self.loss = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)

                self.model.zero_grad()

                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss += loss.item() / args.K
                loss.backward()

                self.optimizer.first_step(zero_grad=True)

                output2 = self.model(input_ids, attention_mask=attention_mask)
                logits2 = output2.logits
                loss2 = self.criterion(logits2, target.long())
                loss2.backward()

                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()),
                                               max_norm=10)
                self.optimizer.second_step(zero_grad=True)

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if
                       'lora' in k or "classifier" in k or "head" in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k or "classifier" in k or "head" in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        return delta_w

    def update_SAMFed(self, weights, E, index, lr, constraint_space=None):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        self.constraint_space = constraint_space
        self.constraint_space_gpu = None
        if constraint_space is not None:
            self.constraint_space_gpu = {
                "D": [[d.to(device) for d in d_list] for d_list in constraint_space["D"]],
                "var": [v.to(device) for v in constraint_space["var"]],
            }

        if args.freeze == 0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True

        base_optimizer = torch.optim.SGD
        trainable_params = [p for _, p in self.model.named_parameters() if p.requires_grad]
        if constraint_space is not None:
            self.optimizer = SAMFed(trainable_params, base_optimizer, lr=lr, weight_decay=0.001, rho=args.rho)
        else:
            self.optimizer = SAM(trainable_params, base_optimizer, lr=lr, weight_decay=0.001, rho=args.rho)

        step = 0
        self.loss = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)

                self.model.zero_grad()

                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss += loss.item() / args.K

                if self.constraint_space_gpu is not None:
                    delta_list, _ = self.solve_restricted_delta(input_ids, attention_mask, target)
                    self.optimizer.first_step(delta_list, zero_grad=True)
                else:
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)

                output2 = self.model(input_ids, attention_mask=attention_mask)
                logits2 = output2.logits
                loss2 = self.criterion(logits2, target.long())
                loss2.backward()

                torch.nn.utils.clip_grad_norm_(parameters=trainable_params, max_norm=10)
                self.optimizer.second_step(zero_grad=True)

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if
                       'lora' in k or "classifier" in k or "head" in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k or "classifier" in k or "head" in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        return delta_w

    def update_NSAM(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)

        if ps_c is None or ps_c == {}:
            ps_c = {k: torch.zeros_like(v, device='cpu') for k, v in self.model.state_dict().items()}

        if args.freeze == 0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True

        shifted_weights = {k: v.clone() for k, v in weights.items()}
        for k in shifted_weights.keys():
            if k in ps_c:
                shifted_weights[k] = shifted_weights[k] + ps_c[k] * args.gamma
        self.model.load_state_dict(shifted_weights)

        base_optimizer = torch.optim.SGD
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = SAM(trainable_params, base_optimizer, lr=lr, weight_decay=0.001, rho=args.rho)

        step = 0
        self.loss = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)

                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss += loss.item() / args.K
                loss.backward()

                self.optimizer.first_step(zero_grad=True)

                output2 = self.model(input_ids, attention_mask=attention_mask)
                logits2 = output2.logits
                loss2 = self.criterion(logits2, target.long())
                loss2.backward()

                torch.nn.utils.clip_grad_norm_(parameters=trainable_params, max_norm=10)
                self.optimizer.second_step(zero_grad=True)

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k or "classifier" in k or "head" in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k or "classifier" in k or "head" in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        return delta_w

    def update_fedavg_adamw(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.01,
                                           betas=(args.beta1, args.beta2))
        step = 0
        self.loss =0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        return delta_w

    def update_FedLADA(self, weights, E, index, momen_m, momen_v, lr, step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if momen_m=={}:
            momen_m = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}

        momen_m2 = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        for k, v in self.model.named_parameters():
            if k not in momen_m.keys():
                continue
            momen_m[k] = momen_m[k].to(device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=lr, weight_decay=0.01)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                self.optimizer.state[p]['step']=step.to(device)
                self.optimizer.state[p]['exp_avg'] = momen_m2[param_name].clone().detach().to(device)
                self.optimizer.state[p]['exp_avg_sq'] = momen_v[param_name].to(device)
        self.loss = 0
        step = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss+=loss.item()/args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if n not in momen_m.keys():
                        continue
                    p.data.add_(momen_m[n].mul(args.gamma)/(args.K))
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                momen_v[param_name] = self.optimizer.state[p]['exp_avg_sq'].to('cpu')

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        return delta_w, momen_v

    def update_FedAdamW(self, weights, E, index, momen_m, momen_v, lr, step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if momen_m=={}:
            momen_m = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}

        momen_m2 = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        for k, v in self.model.named_parameters():
            if k not in momen_m.keys():
                continue
            momen_m[k] = momen_m[k].to(device)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=lr*(1-args.gamma), weight_decay=0.01/(1-args.gamma))
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                self.optimizer.state[p]['step']=step.to(device)
                self.optimizer.state[p]['exp_avg'] = momen_m2[param_name].clone().detach().to(device)
                self.optimizer.state[p]['exp_avg_sq'] =torch.full_like(p.data, momen_v[param_name]).to(device)
        self.loss = 0
        step = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss+=loss.item()/args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if n not in momen_m.keys():
                        continue
                    p.data.add_(momen_m[n].mul(args.gamma)/(args.K))
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                avg_val = self.optimizer.state[p]['exp_avg_sq'].mean()
                momen_v[param_name] = avg_val.to('cpu')

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        return delta_w, momen_v

    def update_fedavg_adam(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.01,
                                           betas=(args.beta1, args.beta2))
        step = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        return delta_w


    def update_FedCM(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}

        self.data_id_loader(index)
        self.gamma = 0.9
        args.gamma=0.9
        self.optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr*(1-args.gamma),
                                         weight_decay=0.001)
        for k, v in self.model.state_dict().items():
            if k in ps_c.keys():
                ps_c[k] = ps_c[k].to(device)
        self.loss = 0
        step = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step=step+1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss+=loss.item()/args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                for n, p in self.model.named_parameters():
                    if not p.requires_grad or n not in ps_c.keys():
                        continue
                    p.data.add_(ps_c[n]*args.gamma/args.K)


        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        return delta_w

    def update_scaf(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        if self.ci == {}:
            self.ci = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001)
        for k in ps_c:
            ps_c[k] = ps_c[k].to(device)
            self.ci[k] = self.ci[k].to(device)
            weights[k] = weights[k].to(device)

        self.loss = 0
        step = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step = step + 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss+=loss.item()/args.K

                lg_loss = 0
                loss_c =  self.criterion(logits, target.long())
                for n, p in self.model.named_parameters():
                    if 'lora' not in n:
                        continue
                    lossh = (p * (-self.ci[n] + ps_c[n])).sum()
                    lg_loss += lossh.item()
                loss = loss_c + lg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        send_ci = {}
        ci = {}
        for k, v in self.model.named_parameters():
            v_cpu = v.detach().to('cpu')
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k] = weights[k].to('cpu')
            ci[k] = self.ci[k]
            self.ci[k] = (weights[k] - v_cpu) / (args.K * lr) + ci[k] - ps_c[k]

        for k, v in self.model.named_parameters():
            if 'lora' not in k:
                continue
            send_ci[k] = -ci[k] + self.ci[k]
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        return delta_w, send_ci





    def update_fedavg_adamw_V(self, weights,momen_v, E, index, lr,step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        self.momen_m = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        if momen_v=={}:
            momen_v= {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}

        self.optimizer = LayerWiseAdamW(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-2,
                                        betas=(args.beta1, args.beta2))
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                self.optimizer.state[p]['step_1'] = int(0)
                self.optimizer.state[p]['step_2'] = int(step)
                self.optimizer.state[p]['exp_avg'] = self.momen_m[param_name].clone().detach().to(device)
                self.optimizer.state[p]['exp_avg_sq'] = momen_v[param_name].clone().detach().to(device)
        step = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                momen_v[param_name] = self.optimizer.state[p]['exp_avg_sq'].clone().detach().to('cpu')
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 10 == 0:
            print(norm)
        return delta_w, momen_v



    def update_FedadamW_CM(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}

        for k, v in self.model.named_parameters():
            if k in ps_c.keys():
                ps_c[k] = ps_c[k].to(device)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr*(1-args.gamma), weight_decay=0.01,
                                           betas=(args.beta1, args.beta2))

        step = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    p.data.add_(-ps_c[n]*args.gamma*lr)

        send_ci = {}
        for k, v in self.model.named_parameters():
            if k in ps_c.keys():
                ps_c[k] = ps_c[k].to('cpu')

        for k, v in self.model.named_parameters():
            if v.requires_grad:
                send_ci[k] = - 1 / (args.K*lr) * (v.cpu() - weights[k])
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 10 == 0:
            print(norm)
        return delta_w,send_ci



    def update_fedavg_adamw_A(self, weights, E, index,momen_v,  lr,step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.momen_m = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                if param_name in self.momen_v.keys():
                    self.optimizer.state[p]['step'] = step.to(device)
                    self.optimizer.state[p]['exp_avg'] =self.momen_m[param_name].clone().detach().to(device)
                    self.optimizer.state[p]['exp_avg_sq'] = momen_v[param_name].clone().detach().to(device)
        step = 0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1
        momen_v={}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                momen_v[param_name] = self.optimizer.state[p]['exp_avg_sq'].clone().detach().to('cpu')
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        return delta_w, momen_v

    def load_dict(self):
        self.func_dict = {
            'FedAvg': self.update_fedavg,
            'FedSAM':self.update_SAM,
            'FedNSAM': self.update_NSAM,
            'SAMFed': self.update_SAMFed,
            'FedAvg_adamw': self.update_fedavg_adamw,
            'FedAvg_adamw_A': self.update_fedavg_adamw_A,
            'FedAvg_adamw_V': self.update_fedavg_adamw_V,
            'FedAvg_adam': self.update_fedavg_adam,
            'FedadamW_CM': self.update_FedadamW_CM,
            'SCAFFOLD': self.update_scaf,
            'FedCM': self.update_FedCM,
            'FedAdamW': self.update_FedAdamW,
            'FedLADA': self.update_FedLADA,
            'FedAdam': self.update_fedavg,
        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None, v=None, step=None, shared_state=None, constraint_space=None):
        self.load_dict()
        if alg in { 'FedCM', 'MoFedSAM', 'FedSWAS',
                   'FedACG','FedLionM','FedadamW_CM','FedCM'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        if alg in {'FedAvg_adamw_A', 'FedAvg_adamw_M','FedAvg_adam_A','FedLADA','FedAdamW'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, v, lr, step)
        if alg in { 'FedAvg_adamw_A','FedAdam_mini_A'}:
            return self.func_dict.get(alg, None)(weights, E, index,  v, lr,step)
        if alg in {'FedAvg_adamw_V','FedAvg_adamw_A','FedAvg_adamw_V2'}:
            return self.func_dict.get(alg, None)(weights, E=E, index=index,  lr=lr, step=step,momen_v=v)
        if alg in { 'FedAvg_adamw_P'}:
            return self.func_dict.get(alg, None)(weights, E, index, lr,shared_state,step)
        if alg in {'SCAFFOLD','SCAFFOLD+'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        if alg in {'FedNSAM'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        if alg in {'SAMFed'}:
            return self.func_dict.get(alg, None)(weights, E, index, lr, constraint_space)
        else:
            return self.func_dict.get(alg, None)(weights, E, index, lr)




def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False











@torch.no_grad()
def apply_weights_adam(num_workers, weights,model,momen_m,momen_v):
    model.to('cpu')
    tau = 0.01
    beta = 0.98
    args.lr_ps=0.01
    train_name=[]
    for name, param in model.named_parameters():
        train_name.append(name)
    delta_t = {}
    for weight in weights:
        for k, v in weight.items():
            if k in delta_t.keys():
                delta_t[k] += v / (num_workers * selection)
            else:
                delta_t[k] = v / (num_workers * selection)
    weight_ps = model.state_dict()
    if momen_m == {}:
        for k, v in delta_t.items():
            momen_m[k] = delta_t[k]*0.1
    else:
        for k, v in delta_t.items():
            momen_m[k] = 0.9 * momen_m[k] + 0.1 * delta_t[k]
    if momen_v == {}:
        momen_v = deepcopy(delta_t)
        for k, v in delta_t.items():
            momen_v[k] = (1 - beta) * v.mul(v)
    else:
        for k, v in momen_v.items():
            momen_v[k] = beta * v + (1 - beta) * delta_t[k].mul(delta_t[k])
    seted_weight = {}
    for k, v in weight_ps.items():
        if k in momen_m.keys():
            seted_weight[k] = (v + args.lr_ps * momen_m[k] / (momen_v[k].sqrt() + tau))
        else:
            seted_weight[k]=v
    model.load_state_dict(seted_weight)
    return model.state_dict(),momen_m,momen_v

def apply_weights_avgACG(num_workers, weights, model, momen_v):
    gamma = args.gamma
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {}
    scale = 1.0 / (num_workers * selection)

    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights:
                sum_weights[k] += v.cpu() * scale
            else:
                sum_weights[k] = v.cpu() * scale

    if not momen_v:
        momen_v = {k: torch.zeros_like(v) for k, v in ps_w.items()}

    for k, v in sum_weights.items():
        if k not in momen_v:
            momen_v[k] = torch.zeros_like(v)
        momen_v[k] = gamma * momen_v[k] + v

    seted_weight = {}
    for k, v in ps_w.items():
        if k in momen_v:
            seted_weight[k] = v + momen_v[k]
        else:
            seted_weight[k] = v

    model.load_state_dict(seted_weight)
    return {k: v.cpu() for k, v in model.state_dict().items()}, momen_v

@torch.no_grad()
def _build_layer_variances(prev_global_weights, current_global_weights, client_params):
    param_keys = list(client_params[0].keys())
    if args.lora == 1:
        param_keys = [k for k in param_keys if ('lora' in k or 'classifier' in k or 'head' in k)]
    else:
        param_keys = [k for k in param_keys if k in prev_global_weights and k in current_global_weights]

    D = []
    var_list = []
    for k in param_keys:
        main_dir = (current_global_weights[k].detach().cpu() - prev_global_weights[k].detach().cpu()).contiguous()
        main_flat = main_dir.reshape(-1)
        main_norm = main_flat.norm() + 1e-12
        main_unit = (main_flat / main_norm).view_as(main_dir).contiguous()

        stacked = torch.stack([
            client_delta[k].detach().cpu()
            for client_delta in client_params
        ], dim=0)
        layer_var = torch.var(stacked, dim=0, unbiased=False)

        D.append([main_unit])
        var_list.append(layer_var.contiguous())
    return D, var_list

@torch.no_grad()
def build_client_constraint_spaces(prev_global_weights, current_global_weights, client_params, budget_ratio=0.5, top_k=4):
    if args.lora == 1:
        filtered_prev = {k: v for k, v in prev_global_weights.items() if ('lora' in k or 'classifier' in k or 'head' in k)}
        filtered_curr = {k: v for k, v in current_global_weights.items() if ('lora' in k or 'classifier' in k or 'head' in k)}
        filtered_clients = [
            {k: v for k, v in client_delta.items() if ('lora' in k or 'classifier' in k or 'head' in k)}
            for client_delta in client_params
        ]
    else:
        filtered_prev = prev_global_weights
        filtered_curr = current_global_weights
        filtered_clients = client_params

    D, var_list = _build_layer_variances(filtered_prev, filtered_curr, filtered_clients)
    return {
        "D": D,
        "var": var_list,
    }

def apply_weights_FedLADA(num_workers, weights,model):
    model.to('cpu')
    m = [mi for _, mi in weights]
    weights = [w for w, _ in weights]
    scale = 1.0 / (num_workers * selection)
    sum_c = {}
    for k, v in m[0].items():
        sum_c[k] =v / (num_workers * selection)
    for ci in m[1:]:
        for k, v in ci.items():
            sum_c[k]+= v / (num_workers * selection)
    ps_w = model.state_dict()
    sum_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():
        ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return model.state_dict(),sum_weights,sum_c


def apply_weights_FedCM(num_workers, weights,model):
    model.to('cpu')
    weights = [w for w in weights]
    sum_c = {}

    ps_w = model.state_dict()
    sum_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():
        ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return model.state_dict(),sum_weights


def apply_weights_SCAF(num_workers, weights,model,ps_c):
    model.to('cpu')
    m = [mi for _, mi in weights]
    weightss = [w for w,_ in weights]
    sum_c = {}
    for k, v in m[0].items():
        sum_c[k] =v / (num_workers * selection)
    for ci in m[1:]:
        for k, v in ci.items():
            sum_c[k]+= v / (num_workers * selection)
    if ps_c == {}:
        ps_c = {k: torch.zeros_like(v.cpu()) for k, v in model.named_parameters()}
        for k, v in m[0].items():
            ps_c[k]=sum_c[k]
    else:
        for k, v in m[0].items():
            if alg in {'SCAFFOLD'}:
                ps_c[k]=ps_c[k]+sum_c[k]*selection
            if alg in {'SCAFFOLD+'}:
                ps_c[k] = ps_c[k] + sum_c[k] * 0.2
    ps_w = model.state_dict()

    sum_weights = {}
    for weight in weightss:
        for k, v in weight.items():
            if k in sum_weights.keys():
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():
        ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return model.state_dict(),ps_c


@torch.no_grad()
def apply_weights_avg(num_workers, weights,model):
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items()}
    scale = 1.0 / (num_workers * selection)
    for weight in weights:
        for k, v in weight.items():
            if 'lora' in k and args.lora==1:
                sum_weights[k].add_(v, alpha=scale)
            else:
                sum_weights[k].add_(v, alpha=scale)
    for k in ps_w.keys():
        ps_w[k].add_(sum_weights[k])
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()}




@torch.no_grad()
def apply_weights_avg2( num_workers,weights,model):
    m = [mi for _, mi in weights]
    weights = [w for w, _ in weights]
    sum_c = {}
    for ci in m:
        for k, v in ci.items():
            if k in sum_c.keys():
                sum_c[k] += v / (num_workers * selection)
            else:
                sum_c[k] = v / (num_workers * selection)
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items()}
    scale = 1.0 / (num_workers * selection)
    for weight in weights:
        for k, v in weight.items():
            if 'lora' in k and args.lora==1:
                sum_weights[k].add_(v, alpha=scale)
            else:
                sum_weights[k].add_(v, alpha=scale)
    for k in ps_w.keys():
        ps_w[k].add_(sum_weights[k])
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()},sum_c


def apply_weights_avg3(num_workers, weights,model):
    m = [mi for _, mi,_ in weights]
    m2 = [vi for _,_, vi in weights]
    weights = [w for w, _,_ in weights]
    scale = 1.0 / (num_workers * selection)
    sum_m = {}
    for k, v in m[0].items():
        sum_m[k] = v / (num_workers * selection)
    for ci in m[1:]:
        for k, v in ci.items():
            sum_m[k]+=v / (num_workers * selection)
    sum_m2 = {}
    for k, v in m2[0].items():
        sum_m2[k] = v.clone().mul_(scale)
    for ci in m2[1:]:
        for k, v in ci.items():
            sum_m2[k].add_(v, alpha=scale)
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {}
    global_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():
        global_weights[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()},sum_m,sum_m2




if __name__ == "__main__":
    gpu_idx = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    set_random_seed(seed=seed)
    epoch = args.epoch
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    E = args.E
    lr_decay = args.lr_decay
    alg = args.alg
    data_name = args.data_name
    selection = args.selection
    tau = args.tau
    lr_ps = args.lr_ps
    alpha_value = args.alpha_value
    alpha = args.alpha
    extra_name = args.extname
    check = args.check
    T_part = args.T_part
    c_dict = {}
    lr_decay = args.lr_decay

    hi_dict = {}
    Li_dict = {}
    mi_dict = {}
    vi_dict = {}
    ti_dict = {}
    import time

    localtime = time.asctime(time.localtime(time.time()))

    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value, extra_name,
                                                                   localtime)
    c_dict = {}
    assert alg in {
        'FedAvg',
        'FedSAM',
        'FedNSAM',
        'SAMFed',
        'FedCM',
        'FedAvg_adamw',
        'FedAdam',
        'FedAvg_adamw_A',
        'FedAvg_adam',
        'FedadamW_CM',
        'FedAvg_adamw_V',
        'FedLADA',
        'SCAFFOLD',
        'FedCM',
        'FedAdamW',
    }
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("./log/{}-{}-{}-{}-{}-{}-{}.txt"
                                  .format(alg, data_name, lr, num_workers, batch_size, E, lr_decay))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(comment=alg)

    nums_cls = 2
    if data_name == 'sst2':
        nums_cls = 2
    if data_name == 'MNLI':
        nums_cls = 3

    nums_sample = 500
    if data_name == 'sst2':
        nums_sample = int(67349/ (args.num_workers))
    if data_name == 'cola':
        nums_sample = int(8551/ (args.num_workers))
    if data_name == 'qnli':
        nums_sample = int(104743/ (args.num_workers))
    if data_name == 'MRPC':
        nums_sample = int(5801/ (args.num_workers))
    if data_name == 'RTE':
        nums_sample = int(2490/ (args.num_workers))
    if data_name == 'MNLI':
        nums_sample = int(392702/ (args.num_workers))
    if data_name == 'QQP':
        nums_sample = int(363846 / (args.num_workers))




    import pickle
    if args.alpha_value == 0.6:
        filename = 'data_idx.data'
    if args.alpha_value == 0.1:
        filename = 'data_idx100000_0.1.data'
    filename = 'data_idx100000_0.1.data'
    if args.alpha_value==1:
        f = open(filename, 'rb')
        data_idx = pickle.load(f)
    else:
        data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)

    ray.init(
        ignore_reinit_error=True,
        num_gpus=num_gpus,
        num_cpus=10,
        object_store_memory=4 * 1024 ** 3,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
    )



    if args.CNN == 'bert':
        model_path = '../glfl/BERT'
        model = BertForSequenceClassification.from_pretrained(model_path)

    if args.CNN == 'roberta_base':
        model_path = './roberta_base'
        model = RobertaForSequenceClassification.from_pretrained(model_path)

        if args.data_name == 'MNLI':
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)



    model=model.to(device)
    epoch_s = 0
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection))]


    logger.info('extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},gamma:{}'
                .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.gamma))

    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")

    if args.CNN == 'bert':
        model_path = '../glfl/BERT'
        model = BertForSequenceClassification.from_pretrained(model_path)

    if args.CNN == 'roberta_base':
        model_path = './roberta_base'
        model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
        if args.data_name == 'MNLI':
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)

        if args.pre==0:
            config = RobertaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=514,
                num_labels=2,
                type_vocab_size=1,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2
            )
            model = RobertaForSequenceClassification(config).to('cpu')
            print('从头训练')


    if args.lora == 1 and args.alg!='FLORA':
        model = get_peft_model(model, lora_config)


    current_weights=model.state_dict()
    ps_c=None

    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    best_acc = 0
    no_improve = 0
    m = {k: torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in
                    model.named_parameters()}
    v = {k: torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in
                    model.named_parameters()}
    if alg in ['FedAvg_adamw_A','FedAdam_mini_A']:
        m = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        v = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        v =None
    momen_m={}
    momen_v = {}
    ps_c={}

    div = []
    sim = []
    step = torch.tensor([0], dtype=torch.float32, device='cpu')
    constraint_space = None
    for epochidx in range(epoch_s, epoch):
        start_time1 = time.time()

        lr = lr * lr_decay

        if args.lr_decay==2:
            eta_max=args.lr
            eta_min=0
            t=epochidx
            T=args.epoch
            lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / T))

        index = np.arange(num_workers)
        np.random.shuffle(index)
        index = index[:int(num_workers * selection)]
        index = np.sort(index)

        if alg in {'FedadamW_CM'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            time3 = time.time()
            current_weights, ps_c = apply_weights_avg2(num_workers, weights_and_ci, model)
            model.load_state_dict(current_weights)




        elif alg in {

            'FedAvg', 'FedSAM', 'FedAdam', 'Fedprox', 'FedIT',
            'FFA_LoRA', 'LoRA_FAIR',
            'FedAvg_adam',
        }:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            current_weights = apply_weights_avg(num_workers, weights,model)
            time4 = time.time()
            model.load_state_dict(current_weights)
        elif alg in {'SAMFed'}:
            if not momen_m:
                momen_m = {k: torch.zeros_like(v, device='cpu') for k, v in current_weights.items()}

            prev_global_weights = {k: v.clone().cpu() for k, v in current_weights.items()}
            index_sel = index
            weights = [
                worker.update_func.remote(alg, current_weights, E, idx, lr, constraint_space=constraint_space)
                for worker, idx in zip(workers, index_sel)
            ]
            weights = ray.get(weights)
            time3 = time.time()
            current_weights, momen_m = apply_weights_avgACG(num_workers, weights, model, momen_m)
            time4 = time.time()
            model.load_state_dict(current_weights)
            constraint_space = build_client_constraint_spaces(prev_global_weights, current_weights, weights)
        elif alg in {'FedNSAM'}:
            if not momen_m:
                momen_m = {k: torch.zeros_like(v, device='cpu') for k, v in current_weights.items()}

            weights = []
            index_sel = index
            weights = [
                worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c=momen_m)
                for worker, idx in zip(workers, index_sel)
            ]
            weights = ray.get(weights)
            time3 = time.time()
            current_weights, momen_m = apply_weights_avgACG(num_workers, weights, model, momen_m)
            time4 = time.time()
            model.load_state_dict(current_weights)

        elif alg in { 'FedAdam' }:
            weights = []
            index_sel = index
            weights = [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                       zip(workers, index_sel)]
            weights = ray.get(weights)
            time3 = time.time()
            current_weights,moment_m = apply_weights_avg(num_workers, weights, model,momen_m)
            time4 = time.time()
            model.load_state_dict(current_weights)



        if alg in {  'FedCM'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            time3 = time.time()
            current_weights, ps_c = apply_weights_FedCM(num_workers, weights_and_ci, model)
            model.load_state_dict(current_weights)
            del weights_and_ci


        if alg in { 'FedadamW_CM'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            time3 = time.time()
            current_weights, ps_c = apply_weights_avg2(num_workers, weights_and_ci, model)
            model.load_state_dict(current_weights)
            del weights_and_ci

        if alg in {'SCAFFOLD','SCAFFOLD+'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [
                    worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c=ps_c)
                    for worker, idx in zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            current_weights, ps_c = apply_weights_SCAF(num_workers, weights_and_ci, model, ps_c)
            model.load_state_dict(current_weights)
            del weights_and_ci

        elif alg in {'FedAdam'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                           zip(workers, index_sel)]
            weights=ray.get(weights)
            current_weights,momen_m,momen_v = apply_weights_adam(num_workers, weights,model,momen_m,momen_v)
            model.load_state_dict(current_weights)
            del weights



        elif alg in { 'FedAvg_adamw'}:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            current_weights = apply_weights_avg(num_workers, weights,model)
            model.load_state_dict(current_weights)



        if alg in {'FedAvg_adamw_A','FedAvg_adamw_V'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [
                    worker.update_func.remote(alg,current_weights, E, idx, lr,v=v, step=step)
                    for
                    worker, idx in
                    zip(workers, index_sel)]
            weights_and_ci=ray.get(weights_and_ci)
            current_weights,v=apply_weights_avg2(num_workers, weights_and_ci,model)
            model.load_state_dict(current_weights)

        if alg in {'FedLADA','FedAdamW'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c=m, v=v,step=step) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            current_weights, m, v = apply_weights_FedLADA(num_workers, weights_and_ci, model)
            model.load_state_dict(current_weights)
            step.add_(args.K)

        end_time1 = time.time()
        print(epochidx, '    ', end_time1 - start_time1)
        args.i = 1

        if epochidx % args.preprint == 0:
            start_time1 = time.time()
            print('测试')
            test_loss = 0
            train_loss = 0
            accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            if epochidx % (args.epoch-1) == 0 and epochidx != 0:
                accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            end_time1 = time.time()
            print('测试完毕', '    ', end_time1 - start_time1)
            test_loss = test_loss.to('cpu')
            loss_train_median = train_loss.to('cpu')
            if accuracy > best_acc:
                best_acc = accuracy
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break

            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info(
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},gamma:{},rho:{},alpha_value:{},data:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.gamma, args.rho, args.alpha_value,
                    args.data_name))

            print(
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},data:{},gamma:{},rho:{},alpha_value:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.data_name, args.gamma,
                    args.rho, args.alpha_value))

            if np.isnan(loss_train_median):
                logger.info('nan~~')
                break
            X_list.append(epochidx)
            result_list.append(accuracy)
            result_list_loss.append(loss_train_median)
            test_list_loss.append(test_loss)

    logger.info("Final accuracy is {:.2f}.".format(accuracy))
    endtime = time.time()
    logger.info('time is pass:{}'.format(endtime - start))
    x = np.array(X_list)
    result = np.array(result_list)
    result_loss = np.array(result_list_loss)
    test_list_loss = np.array(test_list_loss)
    save_name = './plot/alg_{}-data_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-optimizer{}-time{}'.format(
        alg,args.data_name, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        extra_name, args.gamma, args.rho, args.CNN, args.optimizer, endtime)
    save_name = save_name + '.npy'
    np.save(save_name, (x, result, result_loss, test_list_loss))
    ray.shutdown()
