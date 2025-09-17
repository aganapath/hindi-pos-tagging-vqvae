from torch.utils.data import Dataset

class POSDataset(Dataset):
    def __init__(self, sentences, tokenizer, config):
        super().__init__()
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        if self.config.data_mode == "labelled":
            input, tags = zip(*self.sentences[idx])
            input = list(input)
            tags = list(tags)
        else:
            input = self.sentences[idx]
            tags = None

        return self.tokenizer.set_embeddings_mode(input, tags)