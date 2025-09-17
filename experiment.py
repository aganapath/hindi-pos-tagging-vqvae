import os
import shutil
from transformers import AutoModel
from torch.utils.data import DataLoader
from collate_fn import collate_fn
from model_components import *
from embeddings import Embeddings
from tagging_model import TaggingModel
from eval import evaluate
from train import train
from data_vis import plot_training_curves, heatmap, top_gold_words


def experiment(config, experiment_name, decoder, v2i, i2v, l2i, i2l, c2i, i2c, train_dataset, dev_dataset, test_dataset, device):
    save_dir = f"{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)

    loss_values = []
    div_loss_values = []
    vocab_reconstr_values = []
    char_reconstr_values = []
    dev_values = []

    bert_embedding = Embeddings(
        num_chars=len(c2i),
        config=config,
        bert_model=AutoModel.from_pretrained(config.bert_model_name),
        num_layers=1,
        device=device,
        )

    encoder = EncoderFFN(
        emb_hidden=config.emb_hidden,
        num_tag=config.num_tag
        )

    codebook = GumbelCodebook(
        num_tag=config.num_tag,
        tag_dim=config.tag_dim
    )

    vocab_decoder, char_decoder = decoder

    model = TaggingModel(config, bert_embedding, encoder, codebook, vocab_decoder, char_decoder, len(v2i), c2i, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Train model
    loss_over_training, div_loss_over_training, vocab_reconstr_over_training, char_reconstr_over_training, dev_over_training = train(model, train_loader, dev_loader, config, device, experiment_name)
    loss_values.extend(loss_over_training)
    div_loss_values.extend(div_loss_over_training)
    vocab_reconstr_values.extend(vocab_reconstr_over_training)
    char_reconstr_values.extend(char_reconstr_over_training)
    dev_values.extend(dev_over_training)

    print("Training completed!")

    loss_graph = plot_training_curves(experiment_name, loss_values, dev_values, vocab_reconstr_values, div_loss_values)

    accuracy, eval_loss, m1_dict, pred_to_m1, count_dict, gold_words_to_tags_dict = evaluate(model, test_loader, device, config, 'test', l2i, i2l)

    m1_heatmap = heatmap(m1_dict, i2l, experiment_name, m1=True)
    latent_tags_heatmap = heatmap(count_dict, i2l, experiment_name)
    gold_words_df = top_gold_words(config, gold_words_to_tags_dict, i2v, i2l, pred_to_m1, experiment_name)

    shutil.make_archive(f"{experiment_name}", 'zip', save_dir)
    files.download(f"{experiment_name}.zip")

    return model, loss_values, dev_values, div_loss_values, vocab_reconstr_values, char_reconstr_values