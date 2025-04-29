from torch import device
from FlagEmbedding import FlagReranker

def load_reranker_model(model_name: str, device: device) -> FlagReranker:
    if device.type == 'cuda':
        model = FlagReranker(model_name, use_fp16=True, devices=['cuda'])
    else:
        model = FlagReranker(model_name, use_fp16=False, devices=['cpu'])
    return model