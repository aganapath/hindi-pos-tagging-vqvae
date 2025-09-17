import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class Embeddings(nn.Module):
    def __init__(self, num_chars, config, bert_model, device, num_layers=1):
        super().__init__()
        self.config = config
        self.E = self.config.char_dim
        self.R = self.config.word_dim
        self.H = self.config.emb_hidden
        self.W = self.config.max_seq_len
        self.C = self.config.max_word_len
        self.bert_model = bert_model
        self.device = device

        self.cemb = nn.Embedding(num_chars, self.E)
        self.wemb = nn.LSTM(self.E, self.R, num_layers, batch_first=True)
        self.linear = nn.Linear(self.R, self.H)


    def _char_embeddings(self, char_ids, char_word_ids):                        # shape: B, W*C
        char_embeds = self.cemb(char_ids)                                       # shape: B, W*C, E

        pad_mask = einops.rearrange(char_ids == -100, 'B WC -> B WC 1')
        char_embeds = char_embeds.masked_fill(pad_mask, 0.0)

        char_embeds = einops.rearrange(char_embeds, 'B (W C) E -> (B W) C E',   # shape: B*W, C, E
                                       W=self.W, C=self.C)

        _, (hidden, _) = self.wemb(char_embeds)                                 # shape: B*W, R
        word_repr = einops.rearrange(hidden[-1], '(B W) R -> B W R', W=self.W)  # shape: B, W, R

        return self.linear(word_repr)                                           # shape: B, W, H


    def _bert_embeddings(self, token_ids, token_word_ids, attention_mask, special_tokens_mask):
        token2word_mapping = F.one_hot(token_word_ids, num_classes=self.W).to(dtype=torch.float32, device=self.device) #shape: B, T, W

        embeddings = self.bert_model(input_ids=token_ids, attention_mask=attention_mask)
        last_hidden_state = embeddings.last_hidden_state #shape: B, T, H
        last_hidden_state = last_hidden_state.to(self.device)

        content_mask = attention_mask & ~special_tokens_mask
        content_mask = content_mask.to(self.device)

        # Filter to content-only embeddings BEFORE encoding
        content_embeddings = []
        content_lengths = []

        for batch_idx in range(last_hidden_state.size(0)):
            # Get content embeddings for this sequence
            seq_content_mask = content_mask[batch_idx].bool() #makes our mask a T/F; allows more flexibility in how we mask
            seq_content_embeddings = last_hidden_state[batch_idx][seq_content_mask]  # [num_content_tokens, hidden_dim]

            content_embeddings.append(seq_content_embeddings)
            content_lengths.append(seq_content_embeddings.size(0))

        # Pad content embeddings to same length for batching
        padded_content_embeddings = torch.zeros(len(content_embeddings), self.config.max_tok_len, last_hidden_state.size(-1)).to(self.device)
        content_attention_mask = torch.zeros(len(content_embeddings), self.config.max_tok_len).to(self.device)

        for i, (emb, length) in enumerate(zip(content_embeddings, content_lengths)):
            padded_content_embeddings[i, :length] = emb
            content_attention_mask[i, :length] = 1

        return torch.einsum('BTH,BTW->BWH', padded_content_embeddings, token2word_mapping) #shape: B, W, H


    def forward(self, char_ids, char_word_ids, token_ids, token_word_ids, attention_mask, special_tokens_mask):
        bert_embeds = self._bert_embeddings(token_ids, token_word_ids, attention_mask, special_tokens_mask) # shape: B, W, H
        char_embeds = self._char_embeddings(char_ids, char_word_ids) if self.config.use_char_architecture else torch.zeros_like(bert_embeds) # shape: B, W, H

        word_embeds = bert_embeds + char_embeds                                 # shape: B, W, H

        mask_padded_words = torch.where(word_embeds == 0, 0, 1)                 # shape: B, W, D
        word_level_mask = torch.sum(mask_padded_words, dim=-1) > 0              # shape: B, W

        return word_embeds, word_level_mask