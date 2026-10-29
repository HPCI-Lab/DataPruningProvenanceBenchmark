
import os
import torch
from mpi4py import MPI
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup():
    """Initialize the process group for distributed training"""
    
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = rank = comm.Get_rank()

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(world_rank)
    os.environ['LOCAL_RANK'] = "0"

    os.environ['MASTER_PORT'] = '29501'
    os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0' # Frontier specific
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

def cleanup():
    """Clean up the process group after training"""
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

    from mpi4py import MPI
    if MPI.Is_initialized(): MPI.Finalize()

def get_rank():
    return int(os.environ['RANK'])

def get_world_size():
    return int(os.environ['WORLD_SIZE'])

def to_distributed_dataloader(dataset, batch_size):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        pin_memory=True,
        sampler=DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank()),
    )
