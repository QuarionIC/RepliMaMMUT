import cv2
import torch
import numpy as np

def video_to_tensor(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    # Convert list of frames to a NumPy array (T, H, W, C)
    video_np = np.stack(frames)  # shape: (T, H, W, C)

    # Transpose to (C, T, H, W)
    video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2)  # (C, T, H, W)
    return video_tensor

# Example usage
if __name__ == "__main__":
    video_path = "../../MRVD/YouTubeClips/_0nX-El-ySo_83_93.avi"
    tensor = video_to_tensor(video_path)
    print("Video Tensor Shape:", tensor.shape)  # (C, T, H, W)
