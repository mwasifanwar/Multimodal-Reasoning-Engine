import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import TextProcessor, ImageProcessor, AudioProcessor, VideoProcessor, FusionEngine

def basic_demo():
    print("Initializing Multimodal Reasoning Engine...")
    
    text_processor = TextProcessor()
    image_processor = ImageProcessor()
    audio_processor = AudioProcessor()
    video_processor = VideoProcessor()
    
    feature_dims = [768, 2048, 13, 2048]
    fusion_engine = FusionEngine(feature_dims)
    
    sample_text = "A person is walking in a park with a dog"
    print(f"Processing text: {sample_text}")
    
    text_features = text_processor.encode(sample_text)
    image_features = torch.randn(2048)
    audio_features = torch.randn(13)
    video_features = torch.randn(2048)
    
    modalities = [text_features, image_features, audio_features, video_features]
    
    question = "What is happening in the scene?"
    answer = fusion_engine.reason(question, modalities)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("Demo completed successfully!")

if __name__ == "__main__":
    basic_demo()