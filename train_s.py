import argparse
import torch
import wandb
from tqdm import tqdm
from configs.basic_cfg import get_cfg
import os
import torch.distributed as dist
import numpy as np
from utils.datatool import (get_val_transformations,
                            get_train_dataset,
                            get_val_dataset,
                            get_mask_val)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from models.independent_VAE import IVAE
from torch.optim import AdamW, lr_scheduler
from collections import defaultdict
from utils.metrics import clustering_by_representation
from utils.misc import reproducibility_setting
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def smartprint(*msg):
    if LOCAL_RANK ==0 or LOCAL_RANK == -1:
        print(*msg)


def get_device(args, local_rank):
    if args.train.use_ddp:
        device = torch.device(
            f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(f"cuda:{args.train.devices[0]}") if torch.cuda.is_available(
        ) else torch.device('cpu')
    return device

def init_args():
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

@torch.no_grad()
def valid_by_kmeans(val_dataloader, model, use_ddp, device):
    _repr = defaultdict(list)
    targets = []
    for Xs, target in val_dataloader:
        Xs = [x.to(device) for x in Xs]
        if use_ddp:
            spe_repr = model.module.vspecific_features(Xs)
        else:
            spe_repr = model.vspecific_features(Xs)

        for i, r in enumerate(spe_repr):
            _repr[i].append(r)
        targets.append(target)

    targets = torch.concat(targets, dim=-1).numpy()
    result = {}
    for i, key in enumerate(_repr.keys()):
        spe_repr = torch.vstack(_repr[key]).detach().cpu().numpy()
        acc, nmi, ari, _, p, fscore = clustering_by_representation(spe_repr, targets)
        result[f'vspe{i}-acc'] = acc
        result[f'vspe{i}-nmi'] = nmi
        result[f'vspe{i}-ari'] = ari
        result[f'vspe{i}-p'] = p
        result[f'vspe{i}-fscore'] = fscore
    return result


if __name__ == '__main__':
    # load config
    args = init_args()
    config = get_cfg(args.config_file)
    use_wandb = config.wandb
    use_ddp = config.train.use_ddp
    
    result_dir = os.path.join(config.train.log_dir, f"{config.experiment_name}-specific-v{config.vspecific.v_dim}-mv{config.train.mask_view_ratio if config.train.mask_view else 0.0}-{'modal missing' if config.train.val_mask_view else 'full modal'}")
    os.makedirs(result_dir, exist_ok=True)

    if use_ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            [str(i) for i in config.train.devices])
    device = get_device(config, LOCAL_RANK)
    print(f'Use: {device}')

    if use_ddp:
        init_distributed_mode()
    # for reproducibility
    seed = config.seed
    reproducibility_setting(seed)

    checkpoint_path = os.path.join(result_dir, f'checkpoint-{seed}.pth')
    finalmodel_path = os.path.join(result_dir, f'final_model-{seed}.pth')

    val_transformations = get_val_transformations(config)
    train_dataset = get_train_dataset(config, val_transformations)
    #prepare data
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              sampler=train_sampler if use_ddp else None,
                              shuffle=False if use_ddp else True,
                              batch_size=config.train.batch_size,
                              pin_memory=True,
                              drop_last=True)

    #Indepent VAE model
    model = IVAE(args=config, device=device)
    smartprint('model loaded!')

    if use_wandb:
        wandb.init(project=config.project_name,
                   config=config,
                   name=f'{config.experiment_name}-iVAE-c{config.consistency.c_dim}-mv{config.train.mask_view_ratio if config.train.mask_view else 0.0}-{"modal missing" if config.train.mask_view else "full modal"}-{seed}')

    # Only evaluation on the first device
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        if config.train.val_mask_view:
            val_dataset = get_val_dataset(args=config, transform=val_transformations)
        else:
            val_dataset = get_mask_val(config, val_transformations)
        val_dataloader = DataLoader(val_dataset,
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
            broadcast_buffers=False  #模型无缓冲区，减少开销
        )

    best_loss = np.inf
    old_best_model_path = ""
    for epoch in range(config.train.epochs):
        lr = optimizer.param_groups[0]['lr']  # acquire the newest lr
        smartprint("lr:"+str(lr))

        #Train
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)
            model.module.train()
            parameters = list(model.module.parameters())
        else:
            model.train()
            parameters = list(model.parameters())

        cur_loss = defaultdict(list)

        details = {}
        for Xs, _ in tqdm(train_loader):
            Xs = [x.to(device) for x in Xs]
            # assert use_ddp == True
            if use_ddp:
                loss, details = model.module.get_loss(Xs, config.train.masked_ratio, config.train.mask_patch_size)
            else:
                loss, details = model.get_loss(Xs, config.train.masked_ratio, config.train.mask_patch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k, v in details.items():
                cur_loss[k].append(v)

        show_losses = {k: np.mean(v) for k, v in cur_loss.items()}
        if use_wandb:
            wandb.log(show_losses, step=epoch)
        for k, v in show_losses.items():
            smartprint(f"{k}:{v:.4f}")

        smartprint(f"[epoch {epoch}]| Train loss: {loss.item()}")

        # Save the best model
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            if loss <= best_loss:
                # Update best_loss
                best_loss = loss
                best_model_path = os.path.join(result_dir, f"best-{int(loss.item())}-{epoch}-{seed}.pth")
                if old_best_model_path:
                    os.remove(old_best_model_path)
                old_best_model_path = best_model_path

                if use_ddp:
                    model.module.eval()
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    model.eval()
                    torch.save(model.state_dict(), best_model_path)


        # Evaluation of each epoch
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            if use_ddp:
                model.module.eval()
            else:
                model.eval()

            kmeans_result = valid_by_kmeans(val_dataloader, model, use_ddp, device)
            smartprint(f"[Evaluation {epoch}/{config.train.epochs}]", ', '.join([f'{k}:{v:.4f}' for k, v in kmeans_result.items()]))
            if use_wandb:
                wandb.log(kmeans_result)

        #学习率衰减
        if scheduler is not None:
            scheduler.step()
        # Process syn
        if use_ddp:
            dist.barrier()

        
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        final_model_path = os.path.join(result_dir, f"final_model-{config.seed}")
        if use_ddp:
            model.module.eval()
            torch.save(model.module.state_dict(), final_model_path)
        else:
            model.eval()
            torch.save(model.state_dict(), final_model_path)

    if use_ddp:
        dist.destroy_process_group()







            

            











