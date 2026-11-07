import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alg', default='FedRSAM', choices=['FedRSAM', 'FedAvg', 'FedAvgM', 'SCAFFOLD', 'FedSAM', 'FedLESAM', 'FedGAMMA'])
    p.add_argument('--lambda-z', type=float, default=.001)
    p.add_argument('--alpha-lr', type=float, default=.5)
    p.add_argument('--gpu', required=True)
    p.add_argument('--root', default='.')
    p.add_argument('--output', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--rounds', type=int, default=101)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    sys.path.insert(0, a.root)
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from torchvision import transforms
    from peft import LoraConfig, get_peft_model
    from vit_model import vit_base_patch16_224_in21k
    from dataset import TinyImageNetDataset
    from closed_form import finish_delta, build_cone

    torch.set_num_threads(2)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root, out = Path(a.root), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    partition = root / 'num_workers_100-alpha_value_0.01-data_imagenet'
    with partition.open('rb') as f:
        indices = pickle.load(f)
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    train = TinyImageNetDataset(str(root / 'data/tiny-imagenet-200'), split='train', transform=tfm)
    test = TinyImageNetDataset(str(root / 'data/tiny-imagenet-200'), split='test', transform=tfm)
    assert len(train) == 100000 and len(test) == 10000
    assert len(indices) == 100
    for ids in indices.values() if isinstance(indices, dict) else indices:
        assert len(ids) and min(ids) >= 0 and max(ids) < len(train)
    model = vit_base_patch16_224_in21k(num_classes=200)
    weights_path = root / 'vit_base_patch16_224_in21k.pth'
    weights = torch.load(weights_path, map_location='cpu')
    for k in ('head.weight', 'head.bias'):
        weights.pop(k, None)
    print(model.load_state_dict(weights, strict=False), flush=True)
    del weights
    for name, param in model.named_parameters():
        param.requires_grad_('head' in name or 'pre_logits' in name)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        bias='none', task_type='IMAGE_CLASSIFICATION', target_modules=['attn.qkv', 'attn.proj']))
    for name, param in model.named_parameters():
        if 'head' in name or 'classifier' in name:
            param.requires_grad_(True)
    model.cuda()
    params = [v for v in model.parameters() if v.requires_grad]
    names = [k for k, v in model.named_parameters() if v.requires_grad]

    def vector():
        return torch.cat([v.detach().flatten() for v in params])

    def install(v):
        with torch.no_grad():
            pos = 0
            for param in params:
                param.copy_(v[pos:pos + param.numel()].view_as(param))
                pos += param.numel()

    def add_gradient(v):
        pos = 0
        for param in params:
            if param.grad is not None:
                param.grad.add_(v[pos:pos + param.numel()].view_as(param))
            pos += param.numel()

    def evaluate(limit=None):
        model.eval()
        correct = total = 0
        loss_sum = 0.
        with torch.no_grad():
            for b, (x, y) in enumerate(DataLoader(test, batch_size=32, num_workers=2, pin_memory=True)):
                if limit is not None and b >= limit:
                    break
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                logits = model(x)
                loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
        return 100 * correct / total, loss_sum / total, total

    global_w = vector().clone()
    momentum = torch.zeros_like(global_w)
    c = torch.zeros_like(global_w)
    local_c, previous = {}, {}
    cone_direction = cone_variance = None
    plan = dict(vars(a), model='ViT-B/16 IN21K LoRA rank16 alpha32 dropout0.05 + head',
        clients=100, selection=.1, dirichlet_alpha=.01, local_epochs=1,
        max_steps=50, batch_size=16, lr=.1, lr_schedule='cosine over 101 rounds',
        rho=.1, rho_schedule='constant', weight_decay=.001, clip_norm=10,
        server_momentum=.85 if a.alg in ('FedAvgM', 'FedRSAM') else 0., seed=a.seed,
        regularizer_solver='closed_form', alpha_scope='global', cone_scope='global',
        alpha_init=.1, inner_steps=1, client_lookahead_gamma=0., first_round_rho=.1,
        aggregation='uniform selected-client mean', test='TinyImageNet labeled validation 10000',
        trainable_parameters=sum(v.numel() for v in params), trainable_names=names,
        partition_sha256=hashlib.sha256(partition.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        solver_sha256=hashlib.sha256(Path(__file__).with_name('closed_form.py').read_bytes()).hexdigest(),
        lesam_definition='paper: previous received global model minus current, per client; global L2 normalization',
        gamma_definition='FL-Simulator SAM + SCAFFOLD; actual local steps in control update')
    (out / 'config.json').write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan), flush=True)
    selected_rng = np.random.RandomState(a.seed)
    criterion = nn.CrossEntropyLoss()
    fields = ['round', 'lr', 'accuracy_pct', 'test_loss', 'eval_samples', 'train_loss',
              'train_seconds', 'eval_seconds', 'elapsed_seconds', 'local_steps', 'update_norm']
    fields += ['cf_steps', 'alpha_raw_mean', 'residual_norm_mean', 'residual_ratio_mean',
               'perturb_norm_mean', 'box_saturation_mean']
    best = -1.
    rounds = 2 if a.smoke else a.rounds
    with (out / 'metrics.csv').open('w', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in range(rounds):
            tick = time.time()
            lr = .05 * (1 + math.cos(math.pi * t / a.rounds))
            selected = selected_rng.permutation(100)[:10]
            if a.smoke:
                selected = np.array([0, 1])
            updates, losses, step_counts = [], [], []
            diag_total = torch.zeros(5, device='cuda')
            cf_steps = 0
            dc = torch.zeros_like(c)
            for cid0 in selected:
                cid = int(cid0)
                install(global_w)
                model.train()
                opt = torch.optim.SGD(params, lr=lr, weight_decay=.001)
                old_ci = local_c.get(cid, torch.zeros(global_w.numel())).cuda()
                correction = c - old_ci
                perturb = None
                if a.alg == 'FedLESAM':
                    if cid in previous:
                        d = previous[cid].cuda() - global_w
                        perturb = .1 * d / (d.norm() + 1e-12)
                    previous[cid] = global_w.cpu().clone()
                generator = torch.Generator().manual_seed(a.seed + t * 100 + cid)
                loader = DataLoader(train, batch_size=16,
                    sampler=SubsetRandomSampler(indices[cid], generator=generator),
                    num_workers=2, pin_memory=True, generator=generator)
                count, loss_sum = 0, 0.
                for x, y in loader:
                    if count >= (2 if a.smoke else 50):
                        break
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    saved = None
                    if a.alg == 'FedRSAM' and cone_direction is not None:
                        saved = vector().clone()
                        install(saved + .01 * cone_direction)
                        criterion(model(x), y).backward()
                        grad = torch.cat([v.grad.detach().flatten() if v.grad is not None else torch.zeros_like(v).flatten() for v in params])
                        delta, diag = finish_delta(grad, cone_direction, cone_variance,
                            lambda_z=a.lambda_z, alpha_lr=a.alpha_lr)
                        diag_total.add_(diag)
                        cf_steps += 1
                        install(saved + delta)
                        opt.zero_grad(set_to_none=True)
                    elif a.alg in ('FedSAM', 'FedGAMMA', 'FedRSAM'):
                        criterion(model(x), y).backward()
                        grad = torch.cat([v.grad.detach().flatten() if v.grad is not None else torch.zeros_like(v).flatten() for v in params])
                        saved = vector().clone()
                        install(saved + .1 * grad / (grad.norm() + 1e-12))
                        opt.zero_grad(set_to_none=True)
                    elif perturb is not None:
                        saved = vector().clone()
                        install(saved + perturb)
                    loss = criterion(model(x), y)
                    loss.backward()
                    if saved is not None:
                        install(saved)
                    if a.alg in ('SCAFFOLD', 'FedGAMMA'):
                        add_gradient(correction)
                    torch.nn.utils.clip_grad_norm_(params, 10)
                    opt.step()
                    loss_sum += loss.item()
                    count += 1
                assert count > 0
                update = vector() - global_w
                assert torch.isfinite(update).all(), 'nonfinite client update'
                updates.append(update)
                losses.append(loss_sum / count)
                step_counts.append(count)
                if a.alg in ('SCAFFOLD', 'FedGAMMA'):
                    new_ci = old_ci - c - update / (count * lr)
                    dc.add_(new_ci - old_ci)
                    local_c[cid] = new_ci.cpu()
                print(f'round={t} client={cid} steps={count} loss={losses[-1]:.6f}', flush=True)
            avg = torch.stack(updates).mean(0)
            previous_global = global_w.clone()
            if a.alg in ('FedAvgM', 'FedRSAM'):
                momentum.mul_(.85).add_(avg)
                global_w.add_(momentum)
            else:
                global_w.add_(avg)
            c.add_(dc / 100)
            if a.alg == 'FedRSAM':
                cone_direction, cone_variance = build_cone(global_w - previous_global, updates)
            install(global_w)
            torch.cuda.synchronize()
            train_time = time.time() - tick
            ev_start = time.time()
            accuracy = test_loss = ''
            n_eval = 0
            if a.smoke or t % 10 == 0 or t == rounds - 1:
                accuracy, test_loss, n_eval = evaluate(2 if a.smoke else None)
                best = max(best, accuracy)
            row = dict(round=t, lr=lr, accuracy_pct=accuracy, test_loss=test_loss,
                eval_samples=n_eval, train_loss=sum(losses)/len(losses), train_seconds=train_time,
                eval_seconds=time.time()-ev_start, elapsed_seconds=time.time()-started,
                local_steps=sum(step_counts), update_norm=avg.norm().item())
            row['cf_steps'] = cf_steps
            values = (diag_total / max(cf_steps, 1)).tolist()
            row.update(zip(fields[-5:], values))
            writer.writerow(row)
            print(json.dumps(row), flush=True)
            torch.save(dict(round=t, trainable=global_w.cpu(), momentum=momentum.cpu(),
                c=c.cpu(), local_c=local_c, previous=previous, config=plan,
                cone_direction=cone_direction.cpu() if cone_direction is not None else None,
                cone_variance=cone_variance.cpu() if cone_variance is not None else None), out / 'latest.pt')
    (out / 'completed.json').write_text(json.dumps(dict(best_accuracy_pct=best,
        elapsed_seconds=time.time()-started, rounds=rounds, smoke=a.smoke), indent=2))


if __name__ == '__main__':
    main()
