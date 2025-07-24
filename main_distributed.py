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
import warnings
import pickle
from tqdm import tqdm
import aiohttp
import asyncio
import nest_asyncio
nest_asyncio.apply()
warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained("t5-base")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(PARENT_DIR, "COYO")

async def fetch(session, url, text, idx):
    if text is None:
        return None
    try:
        async with session.get(url, timeout=2) as resp:
            if resp.status == 200:
                content = await resp.read()
                image = Image.open(BytesIO(content)).convert("RGB")
                return image, text
    except Exception:
        return None

async def fetch_valid_pairs(batch):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch(session, item['url'], item['text'], idx)
            for idx, item in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks)
        # results = results[1]

    # Filter out failed fetches
    # print(results)
    valid_samples = []
    for res in results:
        if res:
            valid_samples.append(res)
    return valid_samples

def async_remove_none_fn(batch_data):
    # print(batch_data)
    # urls = [item['url'] for item in batch_data]
    # texts = [item['text'] for item in batch_data]
    loop = asyncio.get_event_loop()
    valid_samples = loop.run_until_complete(fetch_valid_pairs(batch_data))
    # images = asyncio.run(fetch_all(urls))  # async batch fetch
    images, texts = zip(*valid_samples)
    texts = list(texts)
    # print(texts)
    
    # Apply transforms if needed
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # valid_img = None
    # for img in images:
    #     if img:
    #         valid_img = img
    #         break
    # if not valid_img:
    #     return []
            
    images = [transform(img) if img else None for img in images] # TODO: Make sure None doesn't cause issues
    # images = torch.Tensor(images)
    tokenized = tokenizer(
        texts,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True)
    images = torch.stack(images)
    # print(images)
    # print(tokenized)
    # return list(zip(images, texts))
    return images, tokenized

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
        return {"url": url, "text": text}
        # Old Synchronous Code
        # TODO: Handle this warning to clean logs/home/hice1/qmot3/scratch/DlSu2025/lib/python3.11/site-packages/PIL/Image.py:1047: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images
        # potential_path = os.path.join(OUTPUT_DIR, str(self.dataset[idx]['id'].item()) + ".jpg")
        # if (os.path.exists(potential_path)):
        #     try:
        #         image = Image.open(potential_path).convert("RGB")
        #     except Exception:
        #         return None
        # else:
        #     try:
        #         response = requests.get(url, timeout=5)
        #         image = Image.open(BytesIO(response.content)).convert("RGB")
        #         image.save(potential_path)
        #         if self.transform:
        #             image = self.transform(image)
        #         return image, text
        #     except Exception:
        #         return None

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

def validation(model, data, rank):
    
    model.eval()
    device = model.device
    epoch = 0

    val_loss = 0
    val_contrastive_loss = 0
    val_generative_loss = 0
    with tqdm(enumerate(data), total=len(data), desc=f"[GPU {rank}] Epoch {epoch}", dynamic_ncols=True) as pbar:
        for step, batch in enumerate(data):

            # input images, and texts
            imgs = batch[0].type(torch.float32).to(device)
            text = batch[1]['input_ids'].type(torch.long).to(device)
            # Since task is to predict next token, the labels will start form position 1
            text_labels = text[:, 1:].to(device)
            total_loss, contrastive_loss, generative_loss = model(imgs, text, text_labels)
            
            pbar.set_postfix(Step=step, Loss=total_loss.item())
            val_loss += total_loss.detach()
            val_contrastive_loss += contrastive_loss.detach()
            val_generative_loss += generative_loss.detach()
            
        val_loss /= len(data)
        val_contrastive_loss /= len(data)
        val_generative_loss /= len(data)

    return val_loss, val_contrastive_loss, val_generative_loss

def load_snapshot(model, epochs_run, snapshot_folder):
    snapshot = torch.load(snapshot_folder + "snapshot.pt")
    model.load_state_dict(snapshot["MODEL_STATE"])
    epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at epoch {epochs_run}")
    return (model, epochs_run)

def save_snapshot(model, epochs_run, snapshot_folder):
    os.makedirs(snapshot_folder, exist_ok=True)
    snapshot = {}
    snapshot["MODEL_STATE"] = model.module.state_dict()
    snapshot["EPOCHS_RUN"] = epochs_run
    torch.save(snapshot, snapshot_folder + "snapshot.pt")
    print(f"Epoch {epochs_run} Training snapshot saved at snapshot.pt")

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def log_batch(model, epochs_run, total_loss, generative_loss, contrastive_loss, norm_type=2):
    print("-----------------------------------------------------------")
    print(f"Epoch: {epochs_run + 1}   Total Loss: {total_loss.item()}   Gen Loss: {generative_loss.item()}   Contr Loss: {contrastive_loss.item()}")
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            grad_norm = param.grad.norm(norm_type).item()
            print(f"{name}: grad norm = {grad_norm:.6f}")

def main(train_batch_size = 64, val_batch_size = 64):
    setup()
    rank = int(os.environ["LOCAL_RANK"])
    snapshot_folder = "./snapshot/"
    epochs_run = 0
    print_logs_every_batch = False

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
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler, collate_fn=async_remove_none_fn)

    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=async_remove_none_fn)
    
    model = MaMMUT(vocab_size=tokenizer.vocab_size, device=rank).to(rank)
    if (os.path.exists(snapshot_folder + "snapshot.pt")):
        model, epochs_run = load_snapshot(model, epochs_run=epochs_run, snapshot_folder=snapshot_folder)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    train(model, train_loader, val_loader, rank, snapshot_folder=snapshot_folder, print_logs_every_batch=print_logs_every_batch, train_batch_size=train_batch_size, val_batch_size = val_batch_size)

    cleanup()

def train(model, data, val_data, rank, lr=0.001, weight_decay=0.000001, num_epochs=20,
          save_every = 1, snapshot_folder = "./snapshots/", epochs_run = 0,
          print_logs_every_batch = False, train_batch_size = 64, val_batch_size = 64):
        
    device = rank
    model.module.train()
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    train_losses = []
    train_contrastive_losses = []
    train_generative_losses = []
    
    val_losses = []
    val_contrastive_losses = []
    val_generative_losses = []

    for epoch in range(epochs_run, num_epochs):
        data.sampler.set_epoch(epoch)
        t_loss = 0
        t_contrastive_loss = 0
        t_generative_loss = 0
        with tqdm(enumerate(data), total=len(data), desc=f"[GPU {rank}] Epoch {epoch}", dynamic_ncols=True) as pbar:
            for step, batch in pbar:
                if not batch: continue
                imgs = batch[0].float().to(device)
                text = batch[1]['input_ids'].long().to(device)
                text_labels = text[:, 1:].to(device)
                total_loss, contrastive_loss, generative_loss = model(imgs, text, text_labels)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=5)
                
                if(print_logs_every_batch):
                    log_batch(model,epochs_run, total_loss, generative_loss, contrastive_loss)
                
                optimizer.step()
                optimizer.zero_grad()
                if rank == 0:
                    pbar.set_postfix(Step=step, Loss=total_loss.item())
                t_loss += total_loss.detach()
                t_contrastive_loss += contrastive_loss.detach()
                t_generative_loss += generative_loss.detach()

            if rank == 0 and epoch % save_every == 0:
                save_snapshot(model, epochs_run=epoch, snapshot_folder=snapshot_folder)

            # Validation (only on rank 0 for simplicity)
            if rank == 0:
                val_loss, val_contrastive_loss, val_generative_loss = validation(model, val_data, rank)
                print(f"[GPU {rank}] Validation loss: {val_loss}")
                train_losses.append(t_loss / (len(data) * train_batch_size))
                train_contrastive_losses.append(t_contrastive_loss / len(data) * train_batch_size)
                train_generative_losses.append(t_generative_loss / len(data) * train_batch_size)
                val_losses.append(val_loss / (len(data) * val_batch_size))
                val_contrastive_losses.append(val_contrastive_loss / (len(data) * val_batch_size))
                val_generative_losses.append(val_generative_loss / (len(data) * val_batch_size))
                
                # with open(f"{snapshot_folder}_train_loss.pkl", 'wb') as f:
                #     pickle.dump(train_losses, f)
                # with open(f"{snapshot_folder}_train_cont_loss.pkl", 'wb') as f:
                #     pickle.dump(train_contrastive_losses, f)
                # with open(f"{snapshot_folder}_train_gen_loss.pkl", 'wb') as f:
                #     pickle.dump(train_generative_losses, f)

                # with open(f"{snapshot_folder}_val_loss.pkl", 'wb') as f:
                #     pickle.dump(val_losses, f)
                # with open(f"{snapshot_folder}_val_cont_loss.pkl", 'wb') as f:
                #     pickle.dump(val_contrastive_losses, f)
                # with open(f"{snapshot_folder}_val_gen_loss.pkl", 'wb') as f:
                #     pickle.dump(val_generative_losses, f)
            epochs_run += 1

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("world_size:", world_size)
    main()