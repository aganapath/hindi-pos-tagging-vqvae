import tqdm
import torch
import numpy as np
from eval_utils import pred_to_gold, many_to_one

def evaluate(model, dataloader, device, config, mode, label2index, index2label):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    match_count = 0
    total_count = 0
    num_batches = 0

    # dictionary to store counts of gold words compared to predicted labels
    word_tag_counts_dict = {i: {} for i in range(config.num_tag)}
    # creating a dictionary to store counts of predicted labels compared to gold labels
    if label2index is not None:
        count_dict = {i: {i: 0 for i in range(config.num_tag)} for i in range(len(label2index))}
    else:
        count_dict = None

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating..."):
            token_ids = batch['token_ids'].to(device)
            token_word_ids = batch['token_word_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['spec_tok_mask'].to(device)
            if config.use_char_architecture:
                char_ids = batch['char_ids'].to(device)
                char_word_ids = batch['char_word_ids'].to(device)
            else:
                char_ids = None
                char_word_ids = None

            vocab_ids = batch['vocab_ids'].to(device)
            loss, vocab_reconstr, char_reconstr, div_loss, tag_logits = model.unsupervised_loss(token_ids, token_word_ids, attention_mask, special_tokens_mask, vocab_ids, char_ids, char_word_ids)
            predicted = torch.argmax(tag_logits, -1)

            total_loss += loss.item()
            num_batches += 1

            if mode == 'test':
                # convert vocab_ids and pred to numpy
                vocab_ids_np = vocab_ids.cpu().detach().numpy()
                predicted_np = predicted.cpu().detach().numpy()
                vocab_predicted_joined = np.rec.fromarrays([vocab_ids_np, predicted_np])

                for seq in vocab_predicted_joined:
                    for gold_word_tag in seq:
                        gold_word = int(gold_word_tag[0])
                        pred_tag = int(gold_word_tag[1])

                        if gold_word != -100:
                            word_tag_counts_dict[pred_tag][gold_word] = word_tag_counts_dict[pred_tag].get(gold_word, 0) + 1

                if label2index is not None:
                    print('labelled data!')
                    labels = batch['labels'].to(device)
                    gold = labels


                    # Convert gold to numpy for confusion-like dictionary
                    gold_np = gold.cpu().detach().numpy()
                    gold_predicted_joined = np.rec.fromarrays([gold_np, predicted_np])

                    M1_dict, pred_to_m1 = many_to_one(gold_predicted_joined, config, label2index, index2label)
                    tag_dict = pred_to_gold(gold_predicted_joined, config, label2index)

            else:
                count_dict = None
                word_tag_counts_dict = None
                M1_dict = None
                pred_to_m1 = None
                tag_dict = None

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = match_count / total_count if total_count > 0 else 0

    print(f"Average loss: {avg_loss:.4f}")

    return accuracy, avg_loss, M1_dict, pred_to_m1, tag_dict, word_tag_counts_dict