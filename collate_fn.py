import torch

def collate_fn(batch):
    """Collate function for DataLoader"""
    vocab_ids = torch.stack([item['vocab_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    char_ids = None
    char_word_ids = None

    if batch[0]['char_ids'] is not None:
        char_ids = torch.stack([item['char_ids'] for item in batch])
    if batch[0]['char_word_ids'] is not None:
        char_word_ids = torch.stack([item['char_word_ids'] for item in batch])

    # Pad word_ids and labels
    max_len = 54

    def pad_sequence(seq_list, pad_value):
        return torch.stack([
            torch.cat([s, torch.full((max_len - s.size(0),), pad_value, dtype=torch.long)])
            if s.size(0) < max_len else s[:max_len]
            for s in seq_list
        ])

    token_ids = pad_sequence([item['token_ids'] for item in batch], pad_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], pad_value=0)
    token_word_ids = pad_sequence([item['token_word_ids'] for item in batch], pad_value=0)
    spec_tok_mask = pad_sequence([item['spec_tok_mask'] for item in batch], pad_value=1)

    return {
        "vocab_ids": vocab_ids,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "token_word_ids": token_word_ids,
        "spec_tok_mask": spec_tok_mask,
        "char_ids": char_ids,
        "char_word_ids": char_word_ids,
        "labels": labels
    }