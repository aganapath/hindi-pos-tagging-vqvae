import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderFFN(nn.Module):
    def __init__(self, emb_hidden, num_tag):
        super().__init__()
        self.ffn = nn.Linear(emb_hidden, num_tag)

    def forward(self, x):
        return self.ffn(x)

class GumbelCodebook(nn.Module):
    def __init__(self, num_tag, tag_dim):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_tag, tag_dim))

    def forward(self, logits, temperature=1.0):
        weights = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
        quantized = weights @ self.codebook
        return quantized, weights

class DecoderFFN(nn.Module):
    def __init__(self, tag_dim, vocab_size):
        super().__init__()
        self.ffn = nn.Linear(tag_dim, vocab_size)

    def forward(self, x):
        return self.ffn(x) # shape: B, W, V

class DecoderBiLSTM(nn.Module):
    def __init__(self, tag_dim: int, dec_hidden: int, vocab_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(tag_dim, dec_hidden, num_layers, bidirectional=True, batch_first=True)
        self.output_projection = nn.Linear(dec_hidden*2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (hidden, cell) = self.lstm(x)  # shape: B, W, D
        return self.output_projection(lstm_out)  # shape: B, W, V

class CharDecoder(nn.Module):
    def __init__(self, tag_dim, hidden_size, char_vocab_size, max_char_len, device):
        super().__init__()
        self.input_projection = nn.Linear(tag_dim, hidden_size)
        self.char_lstm = nn.LSTM(char_vocab_size, hidden_size, batch_first=True)
        self.output_projection = nn.Linear(hidden_size, char_vocab_size)
        self.max_char_len = max_char_len
        self.char_vocab_size = char_vocab_size
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, quantized_repr, target_chars=None, teacher_forcing_ratio=0.5):
        B, W, Q = quantized_repr.shape  # Q = tag_dimension
        C = self.max_char_len  # C = max_word_length (chars per word)
        A = self.char_vocab_size  # A = alphabet_size

        initial_hidden = self.input_projection(quantized_repr)
        initial_hidden = einops.rearrange(initial_hidden, 'B W D -> 1 (B W) D')
        initial_cell = torch.zeros_like(initial_hidden)

        # Initialize with START token (assume index 0)
        decoder_input = torch.zeros(B*W, 1, A).to(self.device)
        decoder_input[:, 0, 0] = 1.0

        outputs = []
        hidden_state = (initial_hidden, initial_cell)

        # Autoregressive character generation
        for t in range(C):
            # LSTM forward pass
            lstm_out, hidden_state = self.char_lstm(decoder_input, hidden_state)

            # Project to character probabilities
            char_logits = self.output_projection(lstm_out)  # [B*W, 1, char_vocab_size]
            outputs.append(char_logits)

            # Prepare next input (teacher forcing vs autoregressive)
            if target_chars is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                if t < C - 1:
                    target_chars_flat = einops.rearrange(target_chars, 'B W C -> (B W) C')
                    next_char = target_chars_flat[:, t]  # [B*W]

                    # Create one-hot encoding
                    decoder_input = torch.zeros(B*W, 1, A).to(self.device)
                    # Handle potential invalid indices
                    valid_mask = (next_char >= 0) & (next_char < A)
                    valid_chars = torch.where(valid_mask, next_char, 0)
                    char_indices = einops.rearrange(valid_chars, 'BW 1 -> BW 1 1')
                    decoder_input.scatter_(2, char_indices, 1.0)
            else:
                # Use model prediction
                predicted_char = torch.argmax(char_logits, dim=-1)  # [B*W, 1]
                decoder_input = torch.zeros(B*W, 1, A).to(self.device)
                pred_indices = einops.rearrange(predicted_char, 'BW 1 -> BW 1 1')
                decoder_input.scatter_(2, pred_indices, 1.0)

        # Concatenate all outputs and reshape
        char_logits = torch.cat(outputs, dim=1)  # [B*W, C, A]
        char_logits = einops.rearrange(char_logits, '(B W) C A -> B W C A', B=B, W=W)

        return char_logits