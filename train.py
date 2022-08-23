import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from time import time
from sys import argv
import random
from warmup_scheduler import GradualWarmupScheduler

import comm
import dataset
from sam import SAM
import hyperparameters
from utils import collate_fn
from detectron2_launch import launch
from distributed_sampler_wrapper import DistributedSamplerWrapper
from runtime_augmentations import cutmix_data, mixup_data, mix_criterion

EPOCH = 101
DEBUG = False
MODEL_TYPE = argv[1].lower()
TASK = 'gender'
DATA_DIR = '.' if len(argv) <= 2 else argv[2]
CKPT_DIR = '.' if len(argv) <= 3 else argv[3]
NUM_GPUS_PER_MACHINE = 1

hyperparameter_fn = getattr(hyperparameters, f"{MODEL_TYPE}_hyperparameters")
model_hyperparameters = hyperparameter_fn()
model_hyperparameters['task'] = TASK
model_hyperparameters['model_name'] = MODEL_TYPE
batch_size = model_hyperparameters['batch_size']

num_workers = 0
pin_memory = True
last_time = 0

def reproducibility(seed):
    torch.manual_seed(seed)
    random.seed(seed) # random
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, load_pretrained, datasets, lr, alpha, warmup_epochs, multiplier, weight_decay, cutmix, mixup, ckpt_folder, amp_enabled, **kwargs):
    last_time = time()
    arguments = dict(locals(), **kwargs)

    reproducibility(seed=1)

    cur_rank = comm.get_local_rank()
    torch.cuda.set_device(cur_rank)
    
    net = model(**arguments)
    net.cuda(cur_rank)

    if comm.is_main_process():
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total Parameters: {pytorch_total_params}")
        
    # Loading pretrained model.
    if comm.is_main_process() and len(argv) >= 5:
        checkpoint = torch.load(f"./checkpoints/{ckpt_folder}/{argv[4]}.pt", map_location=cur_rank)
        state = checkpoint['model_state_dict']
        net.load_state_dict(state, strict=True)

    net = DDP(net, device_ids=[cur_rank], broadcast_buffers=False, find_unused_parameters=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = SAM(net.parameters(), optim.Adam, lr=lr, betas=(.9, .999), weight_decay=weight_decay) # SAM + Adam
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, EPOCH) #  SAM: LR scheduling should be applied to the base optimizer
    scheduler_warmup = GradualWarmupScheduler(optimizer.base_optimizer, multiplier=multiplier, total_epoch=warmup_epochs, after_scheduler=cosine)

    # Due to dataset access limitations, you would need to implement get_dataset yourself to return an MRIDataset object.
    trainset_augment = dataset.get_dataset(datasets, [TASK], DATA_DIR, augment=True)[1]
    
    if comm.is_main_process():
        print('Number of examples in each fold')
        print('trainset augment:', trainset_augment.__len__())

    # Get class counts.
    positive = 0 
    negative = 0 
    for i in trainset_augment.list_IDs:
        label = trainset_augment.labels[trainset_augment.list_IDs[i]]
        if label[TASK] == 1:
            positive += 1
        else:
            negative += 1

    # Construct weights for weighted random sampler.
    weights = []
    for i in trainset_augment.list_IDs:
        label = trainset_augment.labels[trainset_augment.list_IDs[i]][TASK]
        weights.append(1.0/positive if label == 1 else 1.0/negative)
    
    sampler = DistributedSamplerWrapper(sampler=WeightedRandomSampler(weights, len(weights)), num_replicas=comm.get_world_size(), rank=comm.get_local_rank(), shuffle=False)
    
    # Dataloaders
    trainloader_augment = torch.utils.data.DataLoader(
        trainset_augment, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, sampler=sampler)
    if comm.is_main_process():
        print('Finished loading dataset', flush=True)

    if comm.is_main_process():
        print('Initiating training', flush=True)

    for epoch in range(EPOCH):
        trainloader_augment.sampler.set_epoch(epoch)
        net.train()
        epoch_loss = torch.tensor(0, device=cur_rank).float()
        
        print(f"Rank {comm.get_local_rank()} is beginning Epoch {epoch} after {(time() - last_time)/60} minutes.", flush=True)
        last_time = time()
        for i, (inputs, labels) in enumerate(trainloader_augment):
            net.train()

            # get the inputs
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            mixing = False

            cuda = torch.cuda.is_available()
            random_result = random.random()
            if random_result < mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha, cuda)
                mixing = True
            elif random_result < mixup + cutmix:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha, cuda)
                mixing = True

            # forward + backward + optimize
            ## forward pass (autocasted)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = net(inputs.float())
                
                if mixing:
                    targets_a = targets_a.squeeze(1).long()
                    targets_b = targets_b.squeeze(1).long()
                
                labels = labels.squeeze(1).long()

                loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam) if mixing else criterion(outputs, labels)
            
            if DEBUG and comm.is_main_process():
                print(f"Rank {cur_rank}'s Epoch {epoch} Local Loss is {loss}.")

            ## Break out if NaN loss.
            if torch.isnan(loss).any():
                print(torch.isnan(inputs))
                print(inputs)
                print(outputs)
                print(f"Breaking out because of nan in GPU {comm.get_local_rank()}")
                return
    
            ## first backward pass (Note: GradScaler cannot be used here as it does not support SAM)
            with net.no_sync():  # To compute SAM while using multiple GPUs - unsupported in nn.DataParallel
                loss.backward()
            epoch_loss += loss.detach()
            optimizer.first_step(zero_grad=True)

            ## SAM: second forward-backward pass
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                second_step_loss = mix_criterion(criterion, net(inputs.float()), targets_a, targets_b, lam) if mixing else criterion(net(inputs.float()), labels)
            second_step_loss.backward()
            optimizer.second_step(zero_grad=True)

            ## To warm up within epoch
            if (i+1) % (len(trainloader_augment)//3) == 0:
                scheduler_warmup.step()

        if epoch % 5 == 0:
            if comm.is_main_process():
                print('==> Saving model ...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f"{ckpt_folder}/ckpt_main_{epoch}.pt")
    print(f"Rank {comm.get_local_rank()} has finished the last Epoch ({epoch}) after {(time() - last_time)/60} minutes.", flush=True)

def main():
    model_hyperparameters['ckpt_folder'] = CKPT_DIR
    train(**model_hyperparameters)

if __name__ == '__main__':
    training_start_time = time()
    launch(
        main,
        num_gpus_per_machine=NUM_GPUS_PER_MACHINE,
        num_machines=1,
        machine_rank=0,
        dist_url='auto',
        args=()
    )
    print(f"==> Finished Training after {(time() - training_start_time)/60} minutes.")
