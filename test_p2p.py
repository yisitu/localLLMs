import os
import torch
import torch.distributed as dist
import datetime

def test_p2p():
    # standard env vars for torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize Process Group (NCCL)
    # timeout set high to catch hangs vs crashes
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))

    print(f"Rank {rank}: NCCL initialized successfully.")

    # Create a tensor
    tensor = torch.ones(1024 * 1024, device=device) * (rank + 1)

    # Perform All-Reduce (This forces P2P usage on NVLink systems)
    print(f"Rank {rank}: Starting All-Reduce...")
    dist.all_reduce(tensor)
    
    print(f"Rank {rank}: All-Reduce finished. Value: {tensor[0].item()} (Expected: {sum(range(1, world_size + 1))})")

    dist.destroy_process_group()

if __name__ == "__main__":
    test_p2p()
