import torch
import torch.nn as nn
from torch.nn import functional as F
from flask import Flask, request, render_template, send_from_directory
import os

app = Flask(__name__)


@app.route("/")
def request_check():
    return render_template('maxtokenform.html')

@app.route('/favicon.ico')
def fav():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico')

batch_size = 16  # Amount of tokens in a batch
block_size = 34  # Amount of batches in a block as per my understanding
max_iters = 500  # Increase me to get a better result
l_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iterators = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

with open('fulltrumptweetsutf.txt', 'r', encoding='utf-8') as f:
    text = f.read()
len(text)

chars = sorted(list(set(text)))  # Create a list of the text and then sort it
vocab_size = len(chars)  # Length of the sorted list
st = {ch: i for i, ch in enumerate(chars)}  # Step to iterate over the characters
it = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [st[c] for c in s]  # Encoder Function
decode = lambda l: ''.join([it[i] for i in l])  # Decoder Function

data = torch.tensor(encode(text), dtype=torch.long)  # Pass encoded (tokenized) data to the tensor function

train_val = int(
    0.9 * len(data))  # 90% of this will be used for training and 10% used to ensure it isn't just copying the data set
train_data = data[:train_val]
val_data = data[train_val:]
train_data[:block_size + 1]

torch.manual_seed(1432) # Set a random seed

def get_batch(split):
    data = train_data if split == 'train' else val_data  #  If training then move into training sequence
    ix = torch.randint(len(data) - block_size, (batch_size,))  # Iterate through all the data in blocks as defined above
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss_of_data():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterators)
        for k in range(eval_iterators):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Call the training function
    return out  # The output


class Head(nn.Module):
    # Single head / Pretty chill and understandable
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)  # These functions multiply the matrix's generated
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))  # Triangulate the matrix so there is a diagonal of only zeros

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (Block Size, Time, Context)
        q = self.query(x)
        weight = q @ k.transpose(-2,
                              -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T) / As we cannnot multiply these we must transpose the matrix
        weight = weight.masked_fill(self.tril[:T, :T] == 0,
                              float('-inf'))  # Convert all zeros in the diagonal into negative infinity
        weight = F.softmax(weight, dim=-1)  # (B, T, T)
        weight = self.dropout(weight)
        # Aggregation of the values
        v = self.value(x)
        out = weight @ v  # (B, T, T) * (B, T, C) -> (B, T, C)
        return out


# After this comment, it is really difficult for me to understand this he didn't really cover this in his video. I haven't even learnt all this in school :(
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)  # Apply softmax
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramM()
m = model.to(device)

# Number of parameters in the TrumpGPT model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters.', 'We not making ChatGPT with this one')

new_model = BigramM()
new_model.load_state_dict(torch.load('trumpgpt.pt', map_location=torch.device('cpu'), weights_only=True))
m = new_model.to(device)
new_model.eval()
context = torch.zeros((1, 1), dtype=torch.long,
                      device=device)

@app.route('/', methods=['POST'])
def generatedtweets():
    if request.method == 'POST':
        max_tokens = request.form['tokens']
        max_tokens = int(max_tokens)
        generatedresult = decode(m.generate(context, max_new_tokens=max_tokens)[0].tolist())
        return render_template('result.html', result=generatedresult)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)