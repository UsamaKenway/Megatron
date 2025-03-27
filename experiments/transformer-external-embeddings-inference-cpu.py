import torch
from sentence_transformers import SentenceTransformer
import os
os.environ["XFORMERS_DISABLE_MEMORY_EFFICIENT_ATTENTION"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch.utils.checkpoint as checkpoint

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_weights = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn_output)

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
        
class TextDataset(Dataset):
    def __init__(self, df, max_len=100):
        self.texts = df['instruct_prompt'].tolist()
        # Reshape embeddings to match expected dimensions
        self.embeddings = [
            torch.nn.functional.pad(
                get_embeddings(text).unsqueeze(0), 
                (0, 0, 0, max_len - 1)
            ) if len(get_embeddings(text).unsqueeze(0)) < max_len 
            else get_embeddings(text).unsqueeze(0)[:max_len]
            for text in self.texts
        ]
        self.tokenized = [
            tokenizer(
                text, 
                return_tensors='pt', 
                padding="max_length", 
                truncation=True, 
                max_length=max_len
            )["input_ids"].squeeze(0) 
            for text in self.texts
        ]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.tokenized[idx]

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, num_layers, vocab_size, max_len=5000):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)
        # Add input projection layer to handle single token embeddings
        self.input_projection = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # Ensure input has correct shape [batch_size, seq_len, d_model]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Project input
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            # x = layer(x)
            x = checkpoint.checkpoint(layer, x)  # Apply gradient checkpointing per layer
        return self.output_layer(x)

    def decode(self, logits):
        token_ids = torch.argmax(logits, dim=-1)
        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]

######################################       
# Move everything to CPU
device = torch.device('cpu')
model = torch.load('model.pth', map_location=torch.device("cpu"), weights_only=False)
embedding_model =  SentenceTransformer("NovaSearch/stella_en_400M_v5", trust_remote_code=True, device='cuda')# .cuda()

def get_embeddings(texts):
    return torch.tensor(embedding_model.encode(texts), dtype=torch.float32)



# texts = ["Hello, world", "How are you?"]
# embeddings = get_embeddings(texts).unsqueeze(0).to(device)
# model.eval()
# output = model(embeddings)

# decoded_texts = model.decode(output)
# print(decoded_texts)


import torch
from typing import List, Union

class TransformerInference:
    def __init__(self, model, tokenizer, embedding_model):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.device = device # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Convert input text to embeddings."""
        if isinstance(text, str):
            text = [text]
        embeddings = torch.tensor(
            self.embedding_model.encode(text), 
            dtype=torch.float32
        )
        return embeddings

    def pad_embeddings(self, embeddings: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        """Pad or truncate embeddings to specified length."""
        batch_size, seq_len, emb_dim = embeddings.shape
        if seq_len < max_len:
            padding = torch.zeros(batch_size, max_len - seq_len, emb_dim, device=embeddings.device)
            return torch.cat([embeddings, padding], dim=1)
        return embeddings[:, :max_len, :]



    # multiple words.
    @torch.no_grad()
    def generate(
        self,
        text: Union[str, List[str]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate text from input prompt.
        """
        # Prepare input
        embeddings = self.get_embeddings(text)
        embeddings = embeddings.unsqueeze(0) if len(embeddings.shape) == 2 else embeddings
        embeddings = embeddings.to(self.device)
        embeddings = self.pad_embeddings(embeddings)
    
        generated_ids = []
    
        for _ in range(max_length):  # Iterate to generate more tokens
            logits = self.model(embeddings)[:, -1, :]  # Only take last token logits
            
            # Apply temperature
            logits = logits / temperature
    
            # Apply top-k filtering
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                min_values = values[..., -1, None]
                logits[logits < min_values] = float('-inf')
    
            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
    
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
    
            # Append generated token
            generated_ids.append(next_token)
    
            # Convert token ID to embeddings and feed back into the model
            token_embedding = self.embedding_model.encode([self.tokenizer.decode(next_token.item())], convert_to_tensor=True)
            token_embedding = token_embedding.unsqueeze(0).to(self.device)
            embeddings = torch.cat([embeddings, token_embedding], dim=1)
    
        # Decode generated tokens
        generated_ids = torch.cat(generated_ids, dim=-1)
        generated_texts = [
            self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            for ids in generated_ids
        ]
    
        return generated_texts

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

# Example usage
if __name__ == "__main__":
    # Assuming model, tokenizer, and embedding_model are already defined
    inference = TransformerInference(model, tokenizer, embedding_model)
    
    # Single text generation
    prompt = "Write a short story about"
    generated_text = inference(prompt)
    print(f"Input: {prompt}")
    print(f"Generated: {generated_text[0]}")
    
    # Batch generation
    prompts = [
        "Write a poem about",
        "Explain how to",
        "Tell me about"
    ]
    generated_texts = inference(
        prompts,
        max_length=150,
        temperature=0.8,
        top_k=40,
        top_p=0.9
    )
    
    for prompt, generated in zip(prompts, generated_texts):
        print(f"\nInput: {prompt}")
        print(f"Generated: {generated}")