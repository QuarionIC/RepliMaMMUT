# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models, torchvision.datasets
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import shutil
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from models.model import MaMMUT
from datasets import load_dataset
from transformers import AutoTokenizer

# %%

# %%
# Only using 1 parquet file. It contains about 5.8m examples
url = 'https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet'

data_files = {"train": url}

pre_train_data = load_dataset("parquet", data_files=data_files, split="train")

# %%
pre_train_data

# %%
pre_train_data[0]['url']

# %%
pre_train_data[0]['text']

# %%
# using only subset of data for now
pre_train_data = pre_train_data.with_format("torch")
test_data = val_data = pre_train_data.select(range(120000, 130000))
val_data = pre_train_data.select(range(100000, 120000))
pre_train_data = pre_train_data.select(range(100000))


# %%
pre_train_data

# %%

tokenizer = AutoTokenizer.from_pretrained("t5-base")

# %%
# TODO: Need to encode the text descriptions, and clean up images
# TODO: Need to create a final dataset with text, and images
# We can create a custom datalaoder that will load images from urls at runtime.


# %%
import requests
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
#     def encode_text(self, example):
#         text = self.tokenizer(example, padding='max_length', max_length=max_seq_len, add_special_tokens=True) # hard-coded max_length for now
#         bos_id = tokenizer.convert_tokens_to_ids("<s>")
#          # add a bos token as well
#         text = {
#             "input_ids": [bos_id] + text["input_ids"],
#             "attention_mask": [1] + text["attention_mask"]
#         }

        return text

# %%
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

# %%
custom_transforms = transforms.Compose([
    transforms.Resize((272, 272)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])
pre_train_dataset_cleaned = PreTrainDataset(pre_train_data, tokenizer= tokenizer, transform=custom_transforms)
val_dataset_cleaned = PreTrainDataset(val_data, tokenizer= tokenizer, transform=custom_transforms)
train_loader = DataLoader(pre_train_dataset_cleaned, batch_size=32, shuffle=True, collate_fn=remove_none_fn)
val_loader = DataLoader(val_dataset_cleaned, batch_size=10, shuffle=True, collate_fn=remove_none_fn)

# %%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
world_size = torch.cuda.device_count()
print("word_size:", world_size)


# %%
def train(model, data, val_data, lr=0.001, weight_decay=0.000001, num_epochs=20, checkpoint_path='../checkpoints/'):

    model.train()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epoch = 0

#     n = 0

    model = model.to(device)
    train_losses = []
    train_contrastive_losses = []
    train_generative_losses = []
    
    val_losses = []
    val_contrastive_losses = []
    val_generative_losses = []
    epochs = []

    batch_size = 16
    n = 0
    while epoch < num_epochs:

        # Using AdamW for now, can try with other optimizers too
       
        optimizer = optim.AdamW(model.parameters(),
                lr=lr,
                weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        t_loss = 0
        t_contrastive_loss = 0
        t_generative_loss = 0

        for step, batch in enumerate(data):
            
#             print(batch[0], len(batch[0]))
            # input images, and texts
            if not batch:
                continue
            imgs = batch[0].type(torch.float32).to(device)
            text = batch[1]['input_ids'].type(torch.long).to(device)
#             print(text)

            if len(imgs) < batch_size:
                # Last batch will have less images, text pairs since it will be the
                # remainder of Total images / batch_size.

                # Adjust the learning rate of the last batch by 
                # (size(last_batch) / batch_size) to account 
                # for the smaller size.
                adj_lr = lr * (len(imgs) / batch_size)
                optimizer = optim.AdamW(model.parameters(),
                    lr=adj_lr,
                    weight_decay=weight_decay)
            # Since task is to predict next token, the labels will start form position 1
            text_labels = text[:, 1:] 
            total_loss, contrastive_loss, generative_loss = model(imgs, text, text_labels)
            
            n += 1
            print("-----------------------------------------------------------")
            print(f"Iter: {n}   Total Loss: {total_loss.item()}   Gen Loss: {generative_loss.item()}   Contr Loss: {contrastive_loss.item()}")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            i = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
                i += 1
                if i > 10:
                    break
            optimizer.step()
            scheduler.step(total_loss)
            optimizer.zero_grad()

            
            # accumulate epoch loss
            t_loss += total_loss.detach()
            t_contrastive_loss += contrastive_loss.detach()
            t_generative_loss += generative_loss.detach()
            del imgs
            del text
            if n % 500 == 0:
                torch.save(model.state_dict(), f"{checkpoint_path}_epoch_{n}")

        # end of epoch


        epoch += 1

        train_losses.append(t_loss / len(data))
        train_contrastive_losses.append(t_contrastive_loss / len(data))
        train_generative_losses.append(t_generative_loss / len(data))

        epochs.append(epoch)

        val_loss, val_contrastive_loss, val_generative_loss = validation(model, val_data)
        val_losses.append(val_loss)
        val_contrastive_losses.append(val_contrastive_loss)
        val_generative_losses.append(val_generative_loss)
        
#         if epoch % 5 == 0: # save model every 5th epoch
        torch.save(model.state_dict(), f"{checkpoint_path}_epoch_{epoch}")
            
        print("Epoch {}:  Train loss: {}   Train Contrastive Loss: {}   Train Generative Loss: {}]".format(epoch, t_loss / len(data), t_contrastive_loss / len(data), t_generative_loss / len(data)))
        print("Epoch {}:  Val loss: {}   Val Contrastive Loss: {}   Val Generative Loss: {}]".format(epoch, val_loss / len(val_data), val_contrastive_loss / len(val_data), val_generative_loss / len(val_data)))

    return train_losses, train_contrastive_losses, train_generative_losses, val_losses, val_contrastive_losses, val_generative_losses
    

# %%
os.chdir("models")

# %%

model = MaMMUT(vocab_size=tokenizer.vocab_size)

# %%
train(model=model, data=train_loader, val_data=val_loader)

# %%
def validation(model, data):
    
    model.eval()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epoch = 0

    model.to(device)

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
    

# %%



