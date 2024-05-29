import datetime
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttentionHead(nn.Module):
    MAX_SEQ_LEN = 1024

    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        
        self.key = nn.Linear(d_model, d_feature)
        self.query = nn.Linear(d_model, d_feature)
        self.value = nn.Linear(d_model, d_feature)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril_mask', torch.tril(torch.ones(self.MAX_SEQ_LEN, self.MAX_SEQ_LEN)).to(device))
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        k = self.key(x)       # (batch, seq_len, d_model) -> (batch, seq_len, d_feature)
        q = self.query(x)   # (batch, seq_len, d_model) -> (batch, seq_len, d_feature)
        v = self.value(x)   # (batch, seq_len, d_model) -> (batch, seq_len, d_feature)

        # (batch, seq_len, d_feature) x (batch, d_feature, seq_len) -> (batch, seq_len, seq_len)
        attention = q @ k.transpose(-2, -1)
        
        # Mask out the upper half of the matrix, so the model can't look into the future
        attention = attention.masked_fill(self.tril_mask[:seq_len, :seq_len] == 0, float('-inf'))

        # Normalize the attention weights by dividing by the square root of the feature dimension
        attention = attention / (d_model ** 0.5)

        if mask is not None:
            # Expand the mask to match the attention shape (batch_size, 1, seq_len)
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            attention = attention.masked_fill(mask, float('-inf'))

        # Use softmax to get the normalized attention weights from 0 to 1
        attention = torch.softmax(attention, dim=-1)

        # Apply dropout to the attention weights
        attention = self.dropout(attention)

        out = attention @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        d_feature = d_model // n_heads

        self.heads = nn.Sequential(*[AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)])

        self.linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.bias

class MLP(nn.Module):
    def __init__(self, d_model, intermediate, dropout=0.1):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        down_proj = self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))

        return self.dropout(down_proj)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = MLP(d_model, d_model * 4)
    def forward(self, x, mask=None):
        hidden_states = x
        residual = hidden_states

        # input normalization
        hidden_states = self.norm1(hidden_states)

        hidden_states = self.attention(hidden_states, mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
    
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.register_buffer('positional_embedding', self._positional_encoding(d_model))
        self.positional_embedding = nn.Embedding(AttentionHead.MAX_SEQ_LEN, d_model)

        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])

        self.ln = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def _positional_encoding(self, d_model):
        pe = torch.zeros(AttentionHead.MAX_SEQ_LEN, d_model)
        position = torch.arange(0, AttentionHead.MAX_SEQ_LEN).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.size()

        mask = x == 50256
        masked_token_count = mask.sum()

        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        x = self.embedding(x) + self.positional_embedding(pos)

        # x = self.layers(x, kwargs={'mask': mask})
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
            return logits, loss

        logits = logits.view(-1, self.vocab_size)
        targets = targets.reshape(-1)

        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, x, max_new_tokens):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = x[:, -AttentionHead.MAX_SEQ_LEN:]     # (batch, seq_len)

                logits, _ = self(x)

                # this preserves the each sequence in the batch, and the vocab logits, but only the last token in each sequence
                logits = logits[:, -1, :]                           # (batch, vocab_size)

                temperature = 0.7
                top_k = 5
                logits = logits / temperature

                if top_k > 0:
                    # Apply top-k filtering
                    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    mask = torch.full_like(logits, float('-inf'))
                    mask.scatter_(1, top_k_indices, top_k_values)
                    logits = mask

                probs = F.softmax(logits, dim=-1)                   # (batch, vocab_size)

                next = torch.multinomial(probs, num_samples=1)  # (batch, 1)

                # if the next token is the padding token, break
                if next.item() == 50256:
                    break

                x = torch.cat([x, next], dim=1)             # (batch, seq_len + 1)


            return x

from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("gpt2", device=device)
vocab_size = tokenizer.vocab_size
training = True

if training:
    model = Encoder(vocab_size, 512, 16, 16)
    # load the model
    # model.load_state_dict(torch.load('model_step_30000.pth'))

    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    batch_count = {}
    from datatrove.pipeline.readers import ParquetReader
    # https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/data

    data_reader = ParquetReader("hf://datasets/HuggingFaceGECLM/REDDIT_comments/data", text_key = "body", shuffle_files=True)()

    token_buffer = []

    def get_batch(batch_size, seq_len):
        batch = None
        seq_len += 1

        while batch is None or batch.size(0) < batch_size:
            doc = next(data_reader)
            text = doc.text
            tokens = tokenizer([text], return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)

            if tokens.size(1) < seq_len:
                # pad the tokens to the required sequence length
                pad = torch.full((1, seq_len - tokens.size(1)), tokenizer.pad_token_id, dtype=torch.long).to(device)
                tokens = torch.cat([tokens, pad], dim=1)
            elif tokens.size(1) % seq_len != 0:
                tokens = tokens[:, :-(len(tokens[0]) % seq_len)]
            
            tokens = tokens.view(-1, seq_len)

            if batch is None:
                # take a subset of the tokens of size batch_size
                batch = tokens[:batch_size]
            else:
                # concatenate the next batch of tokens
                batch = torch.cat([batch, tokens[:batch_size - batch.size(0)]], dim=0)


        x = batch[:, :-1]
        y = batch[:, 1:]

        x = x.to(device)
        y = y.to(device)

        return x, y

    def train(model, optimizer, scheduler = None, checkpoint_dir = "./checkoints",batch_size=128, seq_len=64, n_steps=50000):
        losses = [0] * 100

        for step in tqdm(range(0, n_steps)):
            model.train()
            X, Y = get_batch(batch_size, seq_len)

            logits, loss = model(X, Y)
            losses[step % 100] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            if step % 100 == 0 and step > 0:

                denominator = 100 if step > 100 else step
                train_loss = sum(losses) / denominator

                print(f"Step {step}, loss: {train_loss}")
            
            if step % 1000 == 0 and step > 0:

                idx = torch.tensor([tokenizer.encode("This")], dtype=torch.long, device=device)
                logits = model.generate(idx, max_new_tokens=50)
                sample = tokenizer.decode(logits[0].tolist())

                print("Sample:\n-------------------")
                print(sample)
                # save checkpoint
                torch.save(model.state_dict(), f'{checkpoint_dir}/model_step_{step}.pth')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = f'checkpoints_{time_stamp}'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    try:
        train(model, optimizer, None, checkpoint_dir=checkpoint_dir, batch_size=128, seq_len=64, n_steps=50000)
    except KeyboardInterrupt:
        print("Training interrupted -- Saving model")
        torch.save(model.state_dict(), f"{checkpoint_dir}/model_interrupted.pth")
        sys.exit()

        
    # Save the model
    torch.save(model.state_dict(), 'model.pth')

else:
    model = Encoder(vocab_size, 512, 8, 8)
    model.load_state_dict(torch.load('backup/model_step_25000.pth'))
    model = model.to(device)

    # Generate some text

    idx = torch.tensor([tokenizer.encode(
"""A""")], dtype=torch.long, device=device)
    logits = model.generate(idx, max_new_tokens=64)
    sample = tokenizer.decode(logits[0].tolist())

    print(sample)