import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from models.model import MaMMUT
from transformers import AutoTokenizer
from datasets import load_dataset
from main import PreTrainDataset, remove_none_fn, custom_transforms, validation  # reuse

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("device with rank:", rank)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def ddp_main(rank, world_size):
    setup(rank, world_size)

    # Data Loading
    url = 'https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet'
    data_files = {"train": url}
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    dataset = dataset.with_format("torch")
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    train_subset = dataset.select(range(100000))
    val_subset = dataset.select(range(100000, 120000))

    train_dataset = PreTrainDataset(train_subset, tokenizer, transform=custom_transforms)
    val_dataset = PreTrainDataset(val_subset, tokenizer, transform=custom_transforms)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=remove_none_fn)

    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=remove_none_fn)

    model = MaMMUT(vocab_size=tokenizer.vocab_size).to(rank)
    model = DDP(model, device_ids=[rank])

    print("Calling training loop")
    train(model, train_loader, val_loader, rank)

    cleanup()

def train(model, data, val_data, rank, lr=0.001, weight_decay=0.000001, num_epochs=20, checkpoint_path='../checkpoints/'):
    device = rank
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(num_epochs):
        data.sampler.set_epoch(epoch)
        for step, batch in enumerate(data):
            if not batch: continue
            imgs = batch[0].float().to(device)
            text = batch[1]['input_ids'].long().to(device)
            text_labels = text[:, 1:]
            total_loss, contrastive_loss, generative_loss = model(imgs, text, text_labels)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            if rank == 0 and step % 100 == 0:
                print(f"[GPU {rank}] Epoch {epoch}, Step {step}, Loss: {total_loss.item()}")

        if rank == 0:
            torch.save(model.state_dict(), f"{checkpoint_path}_epoch_{epoch}.pt")

        # Validation (only on rank 0 for simplicity)
        if rank == 0:
            val_loss, _, _ = validation(model, val_data)
            print(f"[GPU {rank}] Validation loss: {val_loss}")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("world_size:", world_size)
    mp.spawn(ddp_main, args=(world_size,), nprocs=world_size, join=True)