import argparse
import torch
from configs.basic_cfg import get_cfg
import os
import torch.distributed as dist
import numpy as np
from utils.datatool import (get_val_transformations,
                            get_train_dataset,
                            get_val_dataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from models.independent_VAE import IVAE
from torch.optim import AdamW, lr_scheduler
import matplotlib.pyplot as plt

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

def reproducibility_setting(seed):
    """
    set the random seed to make sure reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    print('Global seed:', seed)

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

    for Xs, target in val_dataloader:
        Xs = [x.to(device) for x in Xs]







if __name__ == '__main__':
    # load config
    args = init_args()
    config = get_cfg(args.config_file)

    use_ddp = config.train.use_ddp
    result_dir = os.path.join(config.train.log_dir, f"specific-v{config.vspecific.v_dim}")
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
    # Only evaluation on the first device
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        val_dataset = get_val_dataset(args=config, transform=val_transformations)
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
    show_train_loss = []
    show_val_loss = []
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

        cur_loss = []
        details = {}
        for Xs, _ in train_loader:
            Xs = [x.to(device) for x in Xs]
            # assert use_ddp == True
            if use_ddp:
                losses, details = model.module.get_loss(Xs)
            else:
                losses, details = model.get_loss(Xs)
            loss = torch.stack(losses)
            loss = torch.mean(loss, dim=0)
            cur_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        cur_loss = sum(cur_loss) / len(cur_loss)

        show_train_loss.append(cur_loss)
        
        for k, v in details.items():
            smartprint(f"{k}:{v:.4f}")
            

        smartprint(f"[epoch {epoch}]| Train loss: {cur_loss}")

        # Save the best model
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            if cur_loss <= best_loss:
                # Update best_loss
                best_loss = cur_loss
                best_model_path = os.path.join(result_dir, f"best-{int(cur_loss)}-{epoch}-{seed}.pth")
                if old_best_model_path:
                    os.remove(old_best_model_path)
                old_best_model_path = best_model_path

                if use_ddp:
                    model.module.eval()
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    model.eval()
                    torch.save(model.state_dict(), best_model_path)
            # Evaluation
            cur_val_loss = []
            with torch.no_grad():
                for Xs, _ in val_dataloader:
                    Xs = [x.to(device) for x in Xs]
                    if use_ddp:
                        losses, details = model.module.get_loss(Xs)
                    else:
                        losses, details = model.get_loss(Xs)
                    
                    loss = torch.stack(losses)
                    loss = torch.mean(loss, dim=0)
                    cur_val_loss.append(loss.item())
            

            cur_val_loss = sum(cur_val_loss) / len(cur_val_loss)
            smartprint(f'Val loss: {cur_val_loss}')
            show_val_loss.append(cur_val_loss)
            
                    
                    




        # Update learning rate
        # if scheduler is not None:
        #     scheduler.step()


        # Evaluation of each epoch
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            if use_ddp:
                model.module.eval()
            else:
                model.eval()

            kmeans_result = valid_by_kmeans(val_dataloader, model, use_ddp, device)

        # Process syn
        if use_ddp:
            dist.barrier()

    if LOCAL_RANK == 0 or LOCAL_RANK == -1:

        plt.scatter(range(len(show_train_loss)), show_train_loss, color='blue', label='Train loss')
        plt.scatter(range(len(show_val_loss)), show_val_loss, color='red', label='Val loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend()
        plt.savefig('loss.png')








            

            











