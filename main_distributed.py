import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from models.model import MaMMUT
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
tokenizer = AutoTokenizer.from_pretrained("t5-base")
# print("About to import main")
# from main import PreTrainDataset, remove_none_fn, custom_transforms, validation  # reuse

class PreTrainDataset(Dataset):
    def __init__(self, dataset, tokenizer, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        url = self.dataset[idx]['url']  
        text = self.dataset[idx]['text'] # text has already been encoded and padded 
#         text = self.encode_text(text)
        try:
            response = requests.get(url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, text
        except Exception:
            return None

def remove_none_fn(batch):
    batch_without_nones = [item for item in batch if item is not None]
    if not batch_without_nones:
        return []
    if len(batch_without_nones) < len(batch):
        batch_without_nones.extend([batch_without_nones[-1]] * (len(batch)-len(batch_without_nones)))
    images, texts = zip(*batch_without_nones)
    images = torch.stack(images)
    
    tokenized = tokenizer(
        texts,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True)
    return images, tokenized

def validation(model, data):
    
    model.eval()
    device = model.device
    epoch = 0

    val_loss = 0
    val_contrastive_loss = 0
    val_generative_loss = 0
    
    for step, batch in enumerate(data):

        # input images, and texts
        imgs = batch[0].type(torch.float32).to(device)
        text = batch[1]['input_ids'].type(torch.long).to(device)
        # Since task is to predict next token, the labels will start form position 1
        text_labels = text[:, 1:] 
        total_loss, contrastive_loss, generative_loss = model(imgs, text, text_labels)

        val_loss += total_loss.detach()
        val_contrastive_loss += contrastive_loss.detach()
        val_generative_loss += generative_loss.detach()

    return val_loss, val_contrastive_loss, val_generative_loss

def setup(rank, world_size):
    # Add os.environ["MASTER_ADDR"] = "?"
    # Add os.environ["MASTER_PORT"] = "?"
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
    # tokenizer = AutoTokenizer.from_pretrained("t5-base")
    custom_transforms = transforms.Compose([
        transforms.Resize((272, 272)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
    ])

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
    model.module.train()
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=lr, weight_decay=weight_decay)
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
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=5)
            optimizer.step()

            if rank == 0 and step % 100 == 0:
                print(f"[GPU {rank}] Epoch {epoch}, Step {step}, Loss: {total_loss.item()}")

        if rank == 0:
            torch.save(model.module.state_dict(), f"{checkpoint_path}_epoch_{epoch}.pt")

        # Validation (only on rank 0 for simplicity)
        if rank == 0:
            val_loss, _, _ = validation(model, val_data)
            print(f"[GPU {rank}] Validation loss: {val_loss}")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("world_size:", world_size)
    mp.spawn(ddp_main, args=(world_size,), nprocs=world_size, join=True)