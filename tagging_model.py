import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class TaggingModel(nn.Module):
    def __init__(self, config, embeddings, encoder, codebook, vocab_decoder, char_decoder, vocab_size, c2i, device):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.encoder = encoder
        self.codebook = codebook
        self.vocab_decoder = vocab_decoder
        self.char_decoder = char_decoder
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100).to(device)
        self.c2i = c2i
        self.device = device

    def _calculate_diversity_loss(self, label_probs, word_mask):
        avg_label_probs = torch.logsumexp(
            torch.where(word_mask[:,:,None], label_probs, -torch.tensor(float('inf')).to(self.device)),
            dim=(0, 1)
        ) - torch.log(torch.sum(word_mask)) #shape: l

        actual_entropy = F.cross_entropy(avg_label_probs.unsqueeze(0),  # Add batch dim: [1, num_labels]
                                        torch.exp(avg_label_probs).unsqueeze(0),  # Convert to probs and add batch dim
                                        reduction='sum') #calculates the total entropy rather than the average

        max_entropy = torch.log(torch.tensor(label_probs.shape[-1], dtype=torch.float32).to(self.device))
        diversity_loss = max_entropy - actual_entropy #shape: l

        return diversity_loss

    def _create_char_targets(self, char_ids, char_word_ids, char2index):
        """
        Convert flat character sequences to word-aligned character targets
        """
        B = char_ids.size(0)
        W = self.config.max_seq_len
        C = self.config.max_word_len

        target_chars = torch.full((B, W, C), char2index.get("PAD", 0),
                                dtype=torch.long, device=char_ids.device)
        char_mask = torch.zeros((B, W, C), dtype=torch.bool, device=char_ids.device)

        # Fill in actual characters for each word
        for b in range(B):
            char_pos_in_word = {}  # Track position within each word

            for c_idx, (char_id, word_id) in enumerate(zip(char_ids[b], char_word_ids[b])):
                word_id_val = word_id.item()
                char_id_val = char_id.item()

                # Skip padding tokens and invalid word indices
                if (word_id_val >= W or
                    char_id_val == char2index.get("PAD", -100)):
                    continue

                # Initialize position counter for this word
                if word_id_val not in char_pos_in_word:
                    char_pos_in_word[word_id_val] = 0

                char_pos = char_pos_in_word[word_id_val]

                # Add character if within bounds
                if char_pos < C:
                    target_chars[b, word_id_val, char_pos] = char_id_val
                    char_mask[b, word_id_val, char_pos] = True
                    char_pos_in_word[word_id_val] += 1

        return target_chars, char_mask

    def forward(
            self, char_ids, char_word_ids, # if using character-level embeddings
            token_ids, token_word_ids, attention_mask, special_tokens_mask):

        word_embeddings, word_level_mask = self.embeddings(char_ids, char_word_ids, token_ids, token_word_ids, attention_mask, special_tokens_mask)
        enc_logits = self.encoder(word_embeddings)                              # shape: b, w, l
        quantized, weights = self.codebook(                                     # quantized shape: b, w, tag_dim
            enc_logits, self.config.gumbel_temperature
            )
        word_logits = self.vocab_decoder(quantized)                             # shape: b, w, vocab_size
        word_logprobs = F.log_softmax(word_logits, dim=-1)
        char_seq_logprobs = None

        if self.char_decoder is not None:
            char_seq_logits = self.char_decoder(quantized)
            char_seq_logprobs = F.log_softmax(char_seq_logits, dim=-1)

        label_logprobs = F.log_softmax(enc_logits, dim=-1) #shape: b, w, l

        return label_logprobs, word_logprobs, char_seq_logprobs, word_level_mask, weights


    def unsupervised_loss(self, token_ids, token_word_ids, attention_mask, special_tokens_mask, vocab_ids, char_ids, char_word_ids):
        label_probs, word_probs, char_probs, word_mask, pred_tags = self.forward(char_ids, char_word_ids, token_ids, token_word_ids, attention_mask, special_tokens_mask)

        div_loss = self._calculate_diversity_loss(label_probs, word_mask)

        # ensuring that padding tokens in vocab ids are set to -100 to be ignored by loss calculation
        mask_vocab_ids = torch.where(word_mask, vocab_ids, -100)

        vocab_flat = einops.rearrange(mask_vocab_ids, 'B W -> (B W)')
        word_probs_flat = einops.rearrange(word_probs, 'B W V -> (B W) V')
        vocab_reconstr = self.loss_fn(word_probs_flat, vocab_flat)

        # total_loss = reconstruction_loss + (self.config.diversity_weight * diversity_loss) #calculates a total loss that aggregates both the reconstruction loss and the diversity loss (hopefully to prevent codebook collapse)
        char_reconstr = torch.tensor(0.0, device=self.device)
        if self.char_decoder is not None:
            # Create character targets
            target_chars, char_mask = self._create_char_targets(char_ids, char_word_ids, self.c2i)

            # Flatten for loss calculation
            char_probs_flat = einops.rearrange(char_probs, 'B W C A -> (B W C) A')
            target_chars_flat = einops.rearrange(target_chars, 'B W C -> (B W C)')
            char_mask_flat = einops.rearrange(char_mask, 'B W C -> (B W C)')

            # Apply mask: set padded positions to ignore_index
            masked_char_targets = torch.where(char_mask_flat, target_chars_flat, -100)
            char_reconstr = self.loss_fn(char_probs_flat, masked_char_targets)

        # 4. Combined total loss
        total_loss = (
            (self.config.vocab_loss_weight * vocab_reconstr) +
            (self.config.diversity_weight * div_loss) +
            (self.config.char_loss_weight * char_reconstr)
        )

        return total_loss, vocab_reconstr, char_reconstr, div_loss, pred_tags