from torch import device
from sentence_transformers import SentenceTransformer

def load_dense_model(model_name: str, device: device) -> SentenceTransformer:
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.to(device)
    return model