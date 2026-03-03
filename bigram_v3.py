#This script implements Multi headed self attention

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparameters
batch_size=32  #number of independent sequences we'll process in parallel
block_size=8 #maximum context length for prediction
max_iters= 5000 # number of training steps
eval_interval=500 #print losses after every 300 iterations
learning_rate=1e-3
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200 # number of batches used for loss estimation in eval mode
n_embed=32 
torch.manual_seed(1337)


path="./tinyshakespeare/input.txt"
with open(path,'r', encoding='utf-8') as f:
    text=f.read()


# all the unique chars
chars=sorted(list(set(text)))
vocab_size=len(chars)
# mapping from chars to integers
stoi={ch:i for i, ch in enumerate(chars)}
itos={i:ch for i, ch in enumerate(chars)}
 
encode= lambda s: [stoi[c] for c in s]  #converts text to list of integers
decode= lambda l: "".join([itos[i] for i in l]) # converts list of integers into text


data=torch.tensor(encode(text), dtype=torch.long)
n=int(0.9*len(data))  #90% data used for training, 10% for validation
train_data=data[:n]
val_data=data[n:]
block_size=8
train_data[:block_size+1]
x=train_data[:block_size]
y=train_data[1:block_size+1]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data= train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size, )) 
    # batch size number of random offsets
    # ix will be 4 random numbers between 0 and len(data)-block_size
    x=torch.stack([data[i:i+block_size] for i in ix]) 
    #generating chunks for every  i in ix, we stack them up as rows
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device), y.to(device)
    return x,y

# no training (grad calculation disabled)
@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits, loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear(n_embed, head_size, bias=False)
        self.query=nn.Linear(n_embed, head_size, bias=False)
        self.value=nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C=x.shape
        k=self.key(x) # (B, T, C)
        q=self.query(x) # (B, T, C)
        #compute attention scores ("affinities")
        wei=q @ k.transpose(-2,-1)*C**-0.5  # scaled attention
        #(B,T, C) @ (B, T, C) --> (B, T, T)
        wei=wei.masked_fill(self.tril[:T, :T]==0, float("-inf") )
        wei=F.softmax(wei, dim=-1) #(B, T, T)
        # perform the weighted aggregation of the values
        v=self.value(x) #(B, T, C)
        out=wei@v
        return out 
    
class MultiHeadAttention(nn.Module):
    "Multiple heads of self-attention in parallel."
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non linearity"""
    def __init__(self, n_embed):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_embed, n_embed),
        nn.ReLU(),)
    def forward(self, x):
        return self.net(x)

# simple bigram model
# this model predicts next character based only on current character
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table=nn.Embedding(vocab_size, n_embed) 
        # 24 will go and pluck out 24th row, similarly 48th will go and pluck out 48th row 
        # and then arrange them as B,T,C 
        self.position_embedding_table=nn.Embedding(block_size, n_embed)
        self.sa_heads=MultiHeadAttention(4, n_embed//4) # i.e. 4 heads of 8 dimensional self attention
        self.ffwd=FeedForward(n_embed)
        self.lm_head=nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B, T= idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb=self.token_embedding_table(idx) # (B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x=tok_emb+pos_emb # (B, T, C)
        x=self.sa_heads(x) # apply one head of self-attention. (B, T, C)
        x=self.ffwd(x)
        logits=self.lm_head(x) #(B, T, vocab_size)
        
        if targets is None:
            loss=None
        else: 
            B, T, C=logits.shape
            logits=logits.view(B*T, C) 
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        # job of generator is to take (B,T) and generate (B,T)+1, (B,T)+2 and so on 
        # as many max new tokens
        for _ in range(max_new_tokens): 
            #crop idx to the last block_size tokens
            idx_cond=idx[:, -block_size:] #keep only latest context tokens (context length=8)
            #get the predictions
            logits, loss=self(idx_cond) #get predictions
            # focus only on the last time step
            logits=logits[:, -1,:] # (B,C)  
            # take only last token's prediction
            # apply softmax to get probabilities
            probs=F.softmax(logits, dim=-1) #(B,C)
            # sample from the distribution
            idx_next=torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx=torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
            
    
model=BigramLanguageModel() 
model=model.to(device)

#create a Pytorch optimizer
optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once iinn while evaluate the loss on train and val sets
    if iter % eval_interval==0:
        losses=estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb,yb=get_batch('train')
    # evaluate the loss
    logits, loss=model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context=torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context,max_new_tokens=100)[0].tolist()))

# our model doesn't do anything 
# it has just nn.embedding there's no feedforward layer, batch norm layer etc nothing
# we're not calling backward i.e. no backpropogation, torch.nograd() is there 
