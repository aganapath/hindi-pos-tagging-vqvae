class Config:
    def __init__(self):
        self.data_mode = 'labelled'
        self.use_char_architecture = False
        self.bert_model_name = "google/muril-base-cased"
        # self.bert_model_name = "google-bert/bert-base-uncased" <-- for English BERT

        self.max_seq_len = 32
        self.max_tok_len = 54
        self.max_word_len = 10

        self.char_dim = 64
        self.word_dim = 128

        self.num_tag = 100
        self.tag_dim = 50

        self.emb_hidden = 768 # Must match BERT hidden size
        self.dec_hidden = 256
        self.dec_layers = 2

        self.learning_rate = 6e-5
        self.epochs = 10
        self.batch_size = 256

        self.gumbel_temperature = 1
        self.diversity_weight = 0.6
        self.vocab_loss_weight = 1
        self.char_loss_weight = 1-self.vocab_loss_weight