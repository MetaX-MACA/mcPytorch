import torch
import torch.nn as nn
import numpy as np
import transformers
import copy
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "{}/..".format(cur_dir))
from utils import *

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids = None,
        token_type_ids = None,
        position_ids = None,
        inputs_embeds = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


device = "cuda"

model = BertEmbeddings(transformers.BertConfig()).to(device)
dataset_path = cur_dir + "/data/sentiment-classification-2-6920.npy"
dataset = load_data(dataset_path).tolist()

data = [dataset[0]]

input_ids = [item[0] for item in data]
token_type_ids = [item[1] for item in data]
attention_mask = [item[2] for item in data]
label = [item[3] for item in data]

input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(device)
attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
label = torch.tensor(label, dtype=torch.long).to(device)

model.eval()
golden = model(input_ids=input_ids, position_ids=attention_mask, token_type_ids=token_type_ids)

# model_com
ret = None
model_com = torch.compile(copy.deepcopy(model).to(device), mode="max-autotune", backend="inductor")

for i in range(3):
    ret = timed(lambda: model_com(input_ids=input_ids, position_ids=attention_mask, token_type_ids=token_type_ids))
    print(f"Iter: {i+1}")
    print("     forward time(ms): ", ret[1])
    ret = ret[0]

fw_status = check_close(ret, golden)

if fw_status:
    print("##### success")
    exit(0)
else:
    print(f"##### fail")
    exit(1)