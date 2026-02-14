# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
import torch
import torch.nn as nn
from torch.nn import functional as F

path="./tinyshakespeare/input.txt"
with open(path,'r', encoding='utf-8') as f:
    text=f.read()


# all the unique chars
chars=sorted(list(set(text)))
vocab_size=len(chars)
# mapping from chars to integers
stoi={ch:i for i, ch in enumerate(chars)}
itos={i:ch for i, ch in enumerate(chars)}
encode= lambda s: [stoi[c] for c in s] 
decode= lambda l: "".join([itos[i] for i in l])


data=torch.tensor(encode(text), dtype=torch.long)
n=int(0.9*len(data))  #90% data used for training, 10% for validation
train_data=data[:n]
val_data=data[n:]
block_size=8
train_data[:block_size+1]
x=train_data[:block_size]
y=train_data[1:block_size+1]

torch.manual_seed(1337)
batch_size=4 # number of independent sequences we'll process in parallel
block_size=8 # maximum context length for predictions
max_iters=3000
eval_interval=300
learning_rate=1e-2
device="cuda" if torch.cuda.is_available() else 'cpu'
eval_iters=200

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

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table=nn.Embedding(vocab_size, vocab_size) 
        # 24 will go and pluck out 24th row, similarly 48th will go and pluck out 48th row 
        # and then arrange them as B,T,C 

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits=self.token_embedding_table(idx) # (B,T,C)
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
            idx_cond=idx[:, -block_size:]
            #get the predictions
            logits, loss=self(idx_cond)
            # focus only on the last time step
            logits=logits[:, -1,:] # (B,C)
            # apply softmax to get probabilities
            probs=F.softmax(logits, dim=-1) #(B,C)
            # sample from the distribution
            idx_next=torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx=torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
            
    
model=BigramLanguageModel(vocab_size)
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
