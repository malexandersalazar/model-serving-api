from typing import Tuple, Any, Union

from torch import device
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def load_sparse_model(
    model_name: str, device: device
) -> Tuple[Union[PreTrainedTokenizer, PreTrainedTokenizerFast], Union["Unknown", Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model
