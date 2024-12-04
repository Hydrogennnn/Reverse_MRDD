import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from torch.optim import AdamW, lr_scheduler
from configs.basic_cfg import get_cfg
from models.Reverse_MRDD import RMRDD
import wandb
from utils.metrics import clustering_by_representation
from collections import defaultdict
from utils.datatool import (get_val_transformations,
                            get_train_dataset,
                            get_val_dataset,
                            get_mask_val,
                            add_sp_noise)
from utils.misc import reproducibility_setting
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


@torch.no_grad()
def valid_by_kmeans(val_dataloader, model, use_ddp, device, noise_prob=None):
    targets = []
    consist_reprs = []
    vspecific_reprs = defaultdict(list)
    concate_reprs = defaultdict(list)
    for Xs, target in val_dataloader:
        if noise_prob:
            Xs = [add_sp_noise(x, noise_prob).to(device) for x in Xs]
        else:
            # print(Xs[0].shape)
            Xs = [x.to(device) for x in Xs]
        if use_ddp:
            consist_repr_, vspecific_repr_, concate_repr_ = model.module.all_features(Xs)
        else:
            consist_repr_, vspecific_repr_, concate_repr_ = model.all_features(Xs)   # Tensor, list, list

        targets.append(target)
        consist_reprs.append(consist_repr_.detach().cpu())
        # vspecific_reprs.append(vspecific_repr_.detach().cpu())
        # concate_reprs.append(concate_repr_.detach().cpu())
        for i, (si, c_si) in enumerate(zip(vspecific_repr_, concate_repr_)):
            vspecific_reprs[f"s{i}"].append(si.detach().cpu())
            concate_reprs[f"c+s{i}"].append(c_si.detach().cpu())

    targets = torch.concat(targets, dim=-1).numpy()
    consist_reprs = torch.vstack(consist_reprs).detach().cpu().numpy()
    # vspecific_reprs = torch.vstack(vspecific_reprs).detach().cpu().numpy()
    # concate_reprs = torch.vstack(concate_reprs).detach().cpu().numpy()
    result = {}
    acc, nmi, ari, _, p, fscore = clustering_by_representation(consist_reprs, targets)
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore

    for key, spe_repr in vspecific_reprs.items():
        spe_repr = torch.vstack(spe_repr).detach().cpu().numpy()
        acc, nmi, ari, _, p, fscore = clustering_by_representation(spe_repr, targets)
        result[f'{key}-acc'] = acc
        result[f'{key}-nmi'] = nmi
        result[f'{key}-ari'] = ari
        result[f'{key}-p'] = p
        result[f'{key}-fscore'] = fscore
    for key, cat_repr in concate_reprs.items():
        cat_repr = torch.vstack(cat_repr).detach().cpu().numpy()
        acc, nmi, ari, _, p, fscore = clustering_by_representation(cat_repr, targets)
        result[f'{key}-acc'] = acc
        result[f'{key}-nmi'] = nmi
        result[f'{key}-ari'] = ari
        result[f'{key}-p'] = p
        result[f'{key}-fscore'] = fscore
    return result


def get_device(args, local_rank):
    if args.train.use_ddp:
        device = torch.device(
            f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(f"cuda:{args.train.devices[0]}") if torch.cuda.is_available(
        ) else torch.device('cpu')
    return device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args


def init_distributed_mode():
    # set cuda device
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(
        backend='nccl' if dist.is_nccl_available() else 'gloo')


def get_scheduler(args, optimizer):
    """
    Optimize learning rate
    """
    if args.train.scheduler == 'constant':
        return None
    elif args.train.scheduler == 'linear':
        lf = lambda x: (1 - x / args.train.epochs) * (1.0 - 0.1) + 0.1  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        return scheduler
    elif args.train.scheduler == 'consine':
        eta_min = args.train.lr * (args.train.lr_decay_rate ** 3)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.train.epochs // 10, eta_min=eta_min)
    else:
        scheduler = None
    return scheduler


# Only print on main device
def smartprint(*msg):
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        print(*msg)


if __name__ == '__main__':
    args = parse_args()
    config = get_cfg(args.config_file)

    use_ddp = config.train.use_ddp
    use_wandb = config.wandb
    seed = config.seed
    result_dir = os.path.join(config.train.log_dir,
                              f'{config.experiment_name}-disent-v{config.vspecific.v_dim}-c{config.consistency.c_dim}-m{config.train.masked_ratio}-mv{config.train.mask_view_ratio if config.train.mask_view else 0.0}-{seed}')
    os.makedirs(result_dir, exist_ok=True)

    if use_ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            [str(i) for i in config.train.devices])

    device = get_device(config, LOCAL_RANK)
    print(f"Use: {device}")

    if use_ddp:
        init_distributed_mode()

    seed = config.seed
    reproducibility_setting(seed)

    # Load data
    val_transformations = get_val_transformations(config)
    train_dataset = get_train_dataset(config, val_transformations)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              sampler=train_sampler if use_ddp else None,
                              shuffle=False if use_ddp else True,
                              batch_size=config.train.batch_size,
                              pin_memory=True,
                              drop_last=True)

    # Load model
    model = RMRDD(
        config=config,
        specific_encoder_path=config.vspecific.model_path,
        device=device
    )
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        summary(model)
    if use_wandb:
        wandb.init(project=config.project_name,
                config=config,
                name=f'{config.experiment_name}-rmrdd-c{config.consistency.c_dim}--v{config.vspecific.v_dim}-m{config.train.masked_ratio}-mv{config.train.mask_view_ratio if config.train.mask_view else 0.0}-{seed}')
    summary(model)
    smartprint('model loaded!')
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        mask_val_dataset = get_mask_val(args=config, transform=val_transformations)
        val_dataset = get_val_dataset(config, val_transformations)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=config.train.batch_size // WORLD_SIZE,
                                    num_workers=config.train.num_workers,
                                    shuffle=False,
                                    drop_last=False,
                                    pin_memory=True)
        mask_val_dataloader = DataLoader(mask_val_dataset,
                                    batch_size=config.train.batch_size // WORLD_SIZE,
                                    num_workers=config.train.num_workers,
                                    shuffle=False,
                                    drop_last=False,
                                    pin_memory=True)

        smartprint('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=0.0001, betas=[0.9, 0.95])
    scheduler = get_scheduler(config, optimizer)
    model = model.to(device)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=True,
            broadcast_buffers=False  # 模型无缓冲区，减少开销
        )

    best_loss = np.inf
    old_best_model_path = ""

    for epoch in range(config.train.epochs):
        lr = optimizer.param_groups[0]['lr']
        # wandb.log({'lr':lr}, step=epoch)
        smartprint("lr:" + str(lr))

        # Train
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)
            model.module.train()
            model.module.spe_enc.eval()
        else:
            model.train()
            model.spe_enc.eval()

        cur_loss = defaultdict(list)
        for Xs, _ in tqdm(train_loader):
            Xs = [x.to(device) for x in Xs]
            if use_ddp:
                loss, details = model.module.get_loss(Xs)
            else:
                loss, details = model.get_loss(Xs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k, v in details.items():
                cur_loss[k].append(v)

        show_losses = {k: np.mean(v) for k, v in cur_loss.items()}

        if use_wandb:
            wandb.log(show_losses, step=epoch)
        smartprint(f"[Epoch {epoch}] | Train loss:{loss.item()}")
        for k, v in show_losses.items():
            smartprint(f"{k}:{v}")

        # Check on main process
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            if loss <= best_loss:
                best_loss = loss
                best_model_path = os.path.join(result_dir, f"best-{int(loss.item())}-{epoch}-{seed}.pth")
                if use_ddp:
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
                if old_best_model_path:
                    os.remove(old_best_model_path)
                old_best_model_path = best_model_path

        #学习率衰减
        if scheduler is not None:
            scheduler.step()

        # Evaluation
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            if use_ddp:
                model.module.eval()
            else:
                model.eval()

            # validate on full modal
            kmeans_result = valid_by_kmeans(val_dataloader=val_dataloader,
                                            model=model,
                                            device=device,
                                            use_ddp=use_ddp)
            print(f"[Evaluation {epoch}/{config.train.epochs}]",
                  ', '.join([f'{k}:{v:.4f}' for k, v in kmeans_result.items()]))
            if use_wandb:
                wandb.log(kmeans_result, step=epoch)
            # validate on modal missing
            kmeans_result = valid_by_kmeans(val_dataloader=mask_val_dataloader,
                                            model=model,
                                            device=device,
                                            use_ddp=use_ddp)
            print(f"[Modal missing]",
                  ', '.join([f'{k}:{v:.4f}' for k, v in kmeans_result.items()]))
            if use_wandb:
                for k, v in kmeans_result.items():
                    wandb.log({k+"(modal missing)": v}, step=epoch)
            # validate on full modal with Gaussian Noise
            kmeans_result = valid_by_kmeans(val_dataloader=val_dataloader,
                                            model=model,
                                            device=device,
                                            use_ddp=use_ddp,
                                            noise_prob=config.eval.noise_prob)
            print(f"[Data with Noise]",
                  ', '.join([f'{k}:{v:.4f}' for k, v in kmeans_result.items()]))
            if use_wandb:
                for k, v in kmeans_result.items():
                    wandb.log({k+"(with noise)": v}, step=epoch)



        if use_ddp:
            dist.barrier()

    final_model_path = os.path.join(result_dir, f"final_model-{seed}.pth")
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        if use_ddp:
            torch.save(model.module.state_dict(), final_model_path)
        else:
            torch.save(model.state_dict(), final_model_path)