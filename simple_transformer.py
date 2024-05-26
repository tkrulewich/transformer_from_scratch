import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

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
    
    def forward(self, x):
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
    
    def forward(self, x):
        # Apply each head to the input
        # Concatenate the outputs of each head
        # Apply a linear layer
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)
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

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x):
        x = self.norm1(self.attention(x)) + x
        x = self.norm2(self.mlp(x)) + x
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer('positional_embedding', self._positional_encoding(d_model))

        self.layers = nn.Sequential(*[EncoderBlock(d_model, n_heads) for _ in range(n_layers)])

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

        x = self.embedding(x) + self.positional_embedding[:seq_len]
        x = self.layers(x)
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
                top_k = 2
                logits = logits / temperature

                if top_k > 0:
                    # Apply top-k filtering
                    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    mask = torch.full_like(logits, float('-inf'))
                    mask.scatter_(1, top_k_indices, top_k_values)
                    logits = mask

                probs = F.softmax(logits, dim=-1)                   # (batch, vocab_size)

                next = torch.multinomial(probs, num_samples=1)  # (batch, 1)

                x = torch.cat([x, next], dim=1)             # (batch, seq_len + 1)


            return x

class Tokenizer:
    def __init__(self, chars):
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    def char_to_index(self, ch):
        return self.char_to_ix[ch]
    
    def index_to_char(self, ix):
        return self.ix_to_char[ix]

    def encode(self, text):
        return [self.char_to_index(ch) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.index_to_char(ix) for ix in indices])


from datasets import load_dataset

# This takes a few minutes to run, so go grab a tea or coffee while you wait :)
data_files = "https://huggingface.co/datasets/reddit_tifu/resolve/main/data/tifu_all_tokenized_and_filtered.json.gz"
text = load_dataset("json", data_files=data_files, split="train")["selftext"]


tokenizer = AutoTokenizer.from_pretrained("gpt2", device=device)
vocab_size = tokenizer.vocab_size
training = False

if training:
    model = Encoder(vocab_size, 512, 16, 16)

    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(text, test_size=0.1)

    import random
    def get_batch(data, batch_size, seq_len):
        indices = random.sample(range(len(data)), batch_size)

        text = [data[i] for i in indices]
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=seq_len + 1)['input_ids'].to(device)

        x = tokens[:, :-1]
        y = tokens[:, 1:]
        return x, y

    @torch.no_grad()
    def estimate_loss(model, train_data, val_data, seq_len=128, batch_size=64):
        out = {}
        model.eval()
        for data in [(train_data, 'train'), (val_data, 'val')]:
            losses = []
            for step in range(100):
                X, Y = get_batch(data[0], batch_size, seq_len)
                _, loss = model(X, Y)
                losses.append(loss.item())

            out[data[1]] = sum(losses) / len(losses)

        idx = torch.tensor([tokenizer.encode("The")], dtype=torch.long, device=device)
        logits = model.generate(idx, max_new_tokens=50)
        sample = tokenizer.decode(logits[0].tolist())

        out['sample'] = sample

        model.train()
        return out


    def train(model, optimizer, train_data, test_data, batch_size=32, seq_len=256, n_steps=50000):
        for step in tqdm(range(n_steps)):
            model.train()
            X, Y = get_batch(train_data, batch_size, seq_len)

            logits, loss = model(X, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1000 == 0 and step > 0:
                loss_metrics = estimate_loss(model, train_data, test_data, seq_len=seq_len, batch_size=batch_size)
                print(f"Step {step}, Train loss: {loss_metrics['train']}, Val loss: {loss_metrics['val']}")
                print(f"Sample: {loss_metrics['sample']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    try:
        train(model, optimizer, train_data, test_data, n_steps=50000)
    except KeyboardInterrupt:
        time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        save_name = f'model_{time_stamp}.pth'
        print("Training interrupted -- Saving model as ", save_name)
        torch.save(model.state_dict(), save_name)
        sys.exit()

        
    # Save the model
    torch.save(model.state_dict(), 'model.pth')

else:
    model = Encoder(vocab_size, 512, 16, 16)
    model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)

    # Generate some text

    idx = torch.tensor([tokenizer.encode("This")], dtype=torch.long, device=device)
    logits = model.generate(idx, max_new_tokens=256)
    sample = tokenizer.decode(logits[0].tolist())

    print(sample)