from urllib.request import urlretrieve
from transformers import AutoTokenizer
from config import Config
from tokenizer import Tokenizer
from pos_dataset import POSDataset
from model_components import *
from utils import read_conllu_file, item_indexer
from experiment import experiment

# download Con-LLU files containing data
train_hi_url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/refs/heads/master/hi_hdtb-ud-train.conllu'
train_filename = "data/hi_hdtb-ud-train.conllu"
dev_hi_url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/refs/heads/master/hi_hdtb-ud-dev.conllu'
dev_filename = "data/hi_hdtb-ud-dev.conllu"
test_hi_url = 'https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/refs/heads/master/hi_hdtb-ud-test.conllu'
test_filename = "data/hi_hdtb-ud-test.conllu"

train_path, train_headers = urlretrieve(train_hi_url, train_filename)
dev_path, dev_headers = urlretrieve(dev_hi_url, dev_filename)
test_path, test_headers = urlretrieve(test_hi_url, test_filename)

# instantiate configuration
config = Config()

# instantiate device
curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(curr_device)

train_sents, train_labels = read_conllu_file("data/hi_hdtb-ud-train.conllu")
dev_sents, _ = read_conllu_file("data/hi_hdtb-ud-dev.conllu")
test_sents, _ = read_conllu_file("data/hi_hdtb-ud-test.conllu")

list_of_vocab = [word for sentence in train_sents for word, _ in sentence]
unique_chars = list(set(' '.join(list_of_vocab)))

l2i, i2l = item_indexer(train_labels, True)
v2i, i2v = item_indexer(list_of_vocab)
c2i, i2c = item_indexer(unique_chars)
#
# bert_tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
# tokenizer = Tokenizer(v2i, l2i, c2i, config, bert_tokenizer)
#
# train_dataset = POSDataset(train_sents, tokenizer, config)
# dev_dataset = POSDataset(dev_sents, tokenizer, config)
# test_dataset = POSDataset(test_sents, tokenizer, config)
#
# baseline_decoder = DecoderBiLSTM(
#     tag_dim=config.tag_dim,
#     dec_hidden=config.dec_hidden,
#     vocab_size=len(v2i),
#     num_layers=config.dec_layers
# )
#
# ffn_decoder = DecoderFFN(
#     tag_dim=config.tag_dim,
#     vocab_size=len(v2i)
# )
#
# char_decoder = CharDecoder(
#     tag_dim=config.tag_dim,
#     hidden_size=config.dec_hidden,
#     char_vocab_size=len(c2i),
#     max_char_len=config.max_word_len,
#     device=curr_device
# )
#
# # Each of the experiments below runs a specific experiment for the thesis. Run each one as desired.
#
# # Baseline model
# config.use_char_architecture = False
# decoder = (baseline_decoder, None)
# bl_model, bl_loss_values, bl_dev_values, bl_div_loss_values, bl_vocab_reconstr_values, _ = experiment(config, "Baseline", decoder, v2i, i2v, l2i, i2l, c2i, i2c, train_dataset, dev_dataset, test_dataset, curr_device)
#
# # FFN Decoder model
# config.use_char_architecture = False
# decoder = (ffn_decoder, None)
# ffn_model, ffn_loss_values, ffn_dev_values, ffn_div_loss_values, ffn_vocab_reconstr_values, _ = experiment(config, "FFN Decoder", decoder, v2i, i2v, l2i, i2l, c2i, i2c, train_dataset, dev_dataset, test_dataset, curr_device)
#
# # + Char Embeddings model
# config.use_char_architecture = True
# cemb_model, cemb_loss_values, cemb_dev_values, cemb_div_loss_values, cemb_vocab_reconstr_values, _ = experiment(config, "Char Embeddings", decoder, v2i, i2v, l2i, i2l, c2i, i2c, train_dataset, dev_dataset, test_dataset, curr_device)
#
# # + Char Decoder model
# config.use_char_architecture = True
# config.vocab_loss_weight = 0.5
# decoder = (ffn_decoder, char_decoder)
# cdec_model, cdec_loss_values, cdec_dev_values, cdec_div_loss_values, cdec_vocab_reconstr_values, cdec_char_reconstr_values = experiment(config, "Char Decoder", decoder, v2i, i2v, l2i, i2l, c2i, i2c, train_dataset, dev_dataset, test_dataset, curr_device)
