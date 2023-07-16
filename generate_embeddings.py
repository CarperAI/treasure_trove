from abc import ABC
from datasets import (
    load_dataset,
)
from dotenv import load_dotenv
import torch
from typing import Union, List, Dict

from train_labeler import EncoderParams

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
)

load_dotenv(".env")

# https://huggingface.co/bigcode/starencoder/discussions/3
# https://github.com/bigcode-project/bigcode-encoder/blob/master/embedding_sandbox.ipynb


# https://github.com/bigcode-project/bigcode-encoder/blob/master/src/utils.py#L152
def pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pools a batch of vector sequences into a batch of vector global representations.
    It does so by taking the last vector in the sequence, as indicated by the mask.

    Args:
        x (torch.Tensor): Batch of vector sequences with shape [B, T, F].
        mask (torch.Tensor): Batch of masks with shape [B, T].

    Returns:
        torch.Tensor: Pooled version of the input batch with shape [B, F].
    """

    eos_idx = mask.sum(1) - 1
    batch_idx = torch.arange(len(eos_idx), device=x.device)

    mu = x[batch_idx, eos_idx, :]

    return mu


# https://github.com/bigcode-project/bigcode-encoder/blob/master/src/utils.py#L121
def pool_and_normalize(
    features_sequence: torch.Tensor,
    attention_masks: torch.Tensor,
    return_norms: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Temporal pooling of sequences of vectors and projection onto the unit sphere.

    Args:
        features_sequence (torch.Tensor): Inpute features with shape [B, T, F].
        attention_masks (torch.Tensor): Pooling masks with shape [B, T, F].
        return_norms (bool, optional): Whether to additionally return the norms. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: Pooled and normalized vectors with shape [B, F].
    """

    pooled_embeddings = pooling(features_sequence, attention_masks)
    embedding_norms = pooled_embeddings.norm(dim=1)

    normalizing_factor = torch.where(  # Only normalize embeddings with norm > 1.0.
        embedding_norms > 1.0, embedding_norms, torch.ones_like(embedding_norms)
    )

    pooled_normalized_embeddings = pooled_embeddings / normalizing_factor[:, None]

    if return_norms:
        return pooled_normalized_embeddings, embedding_norms
    else:
        return pooled_normalized_embeddings


# https://github.com/bigcode-project/bigcode-encoder/blob/master/src/constants.py


def set_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    output_data = {}
    for k, v in inputs.items():
        output_data[k] = v.to(device)

    return output_data


def prepare_tokenizer(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=True)

    tokenizer.add_special_tokens({"pad_token": EncoderParams.PAD_TOKEN})
    tokenizer.add_special_tokens({"sep_token": EncoderParams.SEPARATOR_TOKEN})
    tokenizer.add_special_tokens({"cls_token": EncoderParams.CLS_TOKEN})
    tokenizer.add_special_tokens({"mask_token": EncoderParams.MASK_TOKEN})
    return tokenizer


def truncate_sentences(
    sentence_list: List[str], maximum_length: Union[int, float]
) -> List[str]:
    truncated_sentences = []

    for sentence in sentence_list:
        truncated_sentences.append(sentence[:maximum_length])

    return truncated_sentences


class StarEncoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.tokenizer = prepare_tokenizer(EncoderParams.base_model_name)
        self.encoder = (
            AutoModel.from_pretrained(
                EncoderParams.base_model_name, use_auth_token=True
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.max_input_len = EncoderParams.max_input_length
        self.maximum_token_len = EncoderParams.max_token_length

    def forward(self, input_sentences):
        inputs = self.tokenizer(
            [
                f"{EncoderParams.CLS_TOKEN}{sentence}{EncoderParams.SEPARATOR_TOKEN}"
                for sentence in input_sentences
            ],
            padding="longest",
            max_length=self.maximum_token_len,
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.encoder(**set_device(inputs, self.device))
        embedding = pool_and_normalize(outputs.hidden_states[-1], inputs.attention_mask)

        return embedding

    def encode(self, input_sentences, batch_size=32, **kwargs):
        truncated_input_sentences = truncate_sentences(
            input_sentences, self.max_input_len
        )

        n_batches = len(truncated_input_sentences) // batch_size + int(
            len(truncated_input_sentences) % batch_size > 0
        )

        embedding_batch_list = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(truncated_input_sentences))

            with torch.no_grad():
                embedding_batch_list.append(
                    self.forward(truncated_input_sentences[start_idx:end_idx])
                    .detach()
                    .cpu()
                )

        input_sentences_embedding = torch.cat(embedding_batch_list)

        return input_sentences_embedding


tokenizer = AutoTokenizer.from_pretrained(
    EncoderParams.base_model_name, max_length=EncoderParams.max_token_length
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("roborovski/phi-1")

device = torch.device("cuda")
model = StarEncoder(device)


def process(x):
    content = x["content"]
    embedding = model.encode(content)
    return {"embedding": embedding}


# process(dataset["train"][0])

processed_dataset = dataset.map(process, batched=True, batch_size=128)
processed_dataset.push_to_hub("roborovski/phi-2-embeddings")
