import torch
import torch.nn as nn
from transformers import BertModel

class VanillaBert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = outputs[:, 0, :]
        logits = self.dense(pooled_output)
        # logits = self.softmax(self.dense(pooled_output))
        return logits