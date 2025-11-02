import cv2
import torch
import numpy as np
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor

class VideoProcessor:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
    
    def extract_frames(self, video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def encode(self, video_path):
        frames = self.extract_frames(video_path)
        frame_features = []
        
        for i, frame in enumerate(frames):
            cv2.imwrite(f"temp_frame_{i}.jpg", frame)
            features = self.image_processor.encode(f"temp_frame_{i}.jpg")
            frame_features.append(features.cpu().numpy())
        
        video_features = np.mean(frame_features, axis=0)
        return torch.tensor(video_features)