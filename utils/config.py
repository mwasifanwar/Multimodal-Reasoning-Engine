class Config:
    TEXT_MODEL = "microsoft/deberta-v3-base"
    IMAGE_MODEL = "resnet50"
    AUDIO_SAMPLE_RATE = 16000
    VIDEO_FRAME_RATE = 16
    FEATURE_DIM = 256
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    MAX_LENGTH = 512
    
    @classmethod
    def to_dict(cls):
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and not callable(value)}