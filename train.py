import torch
import tqdm
import datetime
from eval import evaluate

def train(model, train_loader, val_loader, config, device, experiment_name, batch_counter=None):
    """Main training loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()

    avg_loss_list = []
    avg_div_loss_list = []
    avg_vocab_reconstr_list = []
    avg_char_reconstr_list = []
    val_loss_list = []

    for epoch in range(config.epochs):
        epoch_loss = 0
        epoch_div_loss = 0
        epoch_vocab_loss = 0
        epoch_char_loss = 0
        num_batches = 0

        for i, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")): #, leave=False)):

            if batch_counter is not None and i >= batch_counter:
                break

            token_ids = batch['token_ids'].to(device)
            token_word_ids = batch['token_word_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vocab_ids = batch['vocab_ids'].to(device)
            special_tokens_mask = batch['spec_tok_mask'].to(device)
            if config.use_char_architecture:
                char_ids = batch['char_ids'].to(device)
                char_word_ids = batch['char_word_ids'].to(device)
            else:
                char_ids = None
                char_word_ids = None

            optimizer.zero_grad()

            loss, vocab_reconstr, char_reconstr, div_loss, _ = model.unsupervised_loss(token_ids, token_word_ids, attention_mask, special_tokens_mask, vocab_ids, char_ids, char_word_ids)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_div_loss += div_loss.item()
            epoch_vocab_loss += vocab_reconstr.item()
            epoch_char_loss += char_reconstr.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_div_loss = epoch_div_loss / num_batches
        avg_vocab_reconstr = epoch_vocab_loss / num_batches
        avg_char_reconstr = epoch_char_loss / num_batches
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}, Diversity loss: {avg_div_loss:.4f}, Vocab Recon loss: {avg_vocab_reconstr:.4f}, Char Recon loss: {avg_char_reconstr:.4f}")

        # Validation
        _, val_loss, _, _, _, _ = evaluate(model, val_loader, device, config, 'validation', None, None)
        model.train()
        val_loss_list.append(val_loss)

        if (epoch + 1) % 10 == 0:
            # Save model state dict
            torch.save(model.state_dict(), f"pos_model_epoch_{epoch+1}_{datetime.date.today()}.pt")

        avg_loss_list.append(avg_loss)
        avg_div_loss_list.append(avg_div_loss)
        avg_vocab_reconstr_list.append(avg_vocab_reconstr)
        avg_char_reconstr_list.append(avg_char_reconstr)

    torch.save(model.state_dict(), f"content/{experiment_name}/final model.pt")

    return avg_loss_list, avg_div_loss_list, avg_vocab_reconstr_list, avg_char_reconstr_list, val_loss_list