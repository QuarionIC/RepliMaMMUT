import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class YouTubeVideoDataset(Dataset):
    def __init__(self, root_dir, frames_per_clip=16, resize=(224, 224)):
        self.video_dir = os.path.join(root_dir, 'YouTubeClips')
        self.description_file = os.path.join(root_dir, 'AllVideoDescriptions.txt')
        self.frames_per_clip = frames_per_clip
        self.resize = resize

        self.video_paths = []
        self.descriptions = {}

        # Load descriptions into a dict
        with open(self.description_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
                video_name, description = parts
                self.descriptions[video_name + ".avi"] = description
        

        # Match .avi files to descriptions
        for file_name in os.listdir(self.video_dir):
            if file_name.endswith('.avi') and file_name in self.descriptions.keys():
                self.video_paths.append(file_name)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_name = self.video_paths[idx]
        video_path = os.path.join(self.video_dir, video_name)

        # Load video as tensor (C, T, H, W)
        video_tensor = self._load_video(video_path)

        # Get description
        description = self.descriptions[video_name]

        return video_tensor, description

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames found in {path}")

        # Sample or pad frames
        frames = self._sample_or_pad(frames)

        video_np = np.stack(frames)  # (T, H, W, C)
        video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
        return video_tensor

    def _sample_or_pad(self, frames):
        total = len(frames)
        if total >= self.frames_per_clip:
            indices = np.linspace(0, total - 1, self.frames_per_clip, dtype=int)
            return [frames[i] for i in indices]
        else:
            # Pad with last frame
            last = frames[-1]
            pad = [last] * (self.frames_per_clip - total)
            return frames + pad

# Usage
if __name__ == "__main__":
    root = "/home/hice1/qmot3/scratch/MRVD"
    dataset = YouTubeVideoDataset(root_dir=root)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for videos, captions in dataloader:
        print("Video batch shape:", videos.shape)  # (B, C, T, H, W)
        print("Captions:", captions)
        break
