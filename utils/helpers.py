import logging
import json
import torch
import numpy as np

def setup_logging(name="multimodal_engine"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{name}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def save_results(results, filename="results.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)