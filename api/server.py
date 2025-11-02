from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from core import TextProcessor, ImageProcessor, AudioProcessor, VideoProcessor, FusionEngine

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    text_processor = TextProcessor()
    image_processor = ImageProcessor()
    audio_processor = AudioProcessor()
    video_processor = VideoProcessor()
    
    feature_dims = [768, 2048, 13, 2048]
    fusion_engine = FusionEngine(feature_dims)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "model": "multimodal-reasoning-engine"})
    
    @app.route('/process', methods=['POST'])
    def process_multimodal():
        data = request.json
        
        text_feat = text_processor.encode(data.get('text', ''))
        image_feat = image_processor.encode(data.get('image_path', ''))
        audio_feat = audio_processor.encode(data.get('audio_path', ''))
        video_feat = video_processor.encode(data.get('video_path', ''))
        
        modalities = [text_feat, image_feat, audio_feat, video_feat]
        answer = fusion_engine.reason(data.get('question', ''), modalities)
        
        return jsonify({
            "question": data.get('question', ''),
            "answer": answer,
            "status": "processed"
        })
    
    return app