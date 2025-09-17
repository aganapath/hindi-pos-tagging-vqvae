import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from eval_utils import m1_accuracy
import datetime

def plot_training_curves(
    experiment_name,
    loss_values,
    val_values,
    vocab_loss_values=None,
    div_loss_values=None,
    char_loss_values=None
    ):

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    # Set figure
    plt.figure(figsize=(10, 6))

    # Plot available curves
    plt.plot(np.array(loss_values), 'r', label='Training Loss')
    plt.plot(np.array(val_values), 'b', label='Validation Loss')

    if vocab_loss_values is not None:
        plt.plot(np.array(vocab_loss_values), 'm', label='Vocab Reconstruction Loss')

    if char_loss_values is not None:
        plt.plot(np.array(char_loss_values), 'y', label='Char Reconstruction Loss')

    if div_loss_values is not None:
        div_loss_2 = [loss*10 for loss in div_loss_values]
        plt.plot(np.array(div_loss_2), 'c', label='Diversity Loss (x10)')

    # Labels and title
    plt.xlabel("\nEpoch\n")
    plt.ylabel("\nLoss Value\n")
    title = f"\n{experiment_name} Training & Validation Loss Curves\n"
    plt.title(title)

    # Final touches
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(f"content/{experiment_name}/loss curves.png", dpi=300, bbox_inches='tight')

    np.savez(
    f"content/{experiment_name}/loss arrays.npz",
    loss_values=loss_values,
    val_values=val_values,
    vocab_loss_values=vocab_loss_values,
    div_loss_values=div_loss_values,
    char_loss_values=char_loss_values
    )

    data = {
            "epoch": np.arange(len(loss_values)),
            "loss_values": loss_values,
            "val_values": val_values
        }

    if vocab_loss_values is not None:
        data["recon_loss_values"] = vocab_loss_values
    if div_loss_values is not None:
        data["div_loss_values"] = div_loss_values
    if char_loss_values is not None:
        data["char_loss_values"] = char_loss_values

    df = pd.DataFrame(data)
    df.to_csv(f"content/{experiment_name}/loss data.csv", index=False)

    plt.show()

def heatmap(counts, i2l, experiment_name, m1=False):
    tag_order = [
        'NOUN', 'PROPN', 'PRON', 'NUM',              # Nominal Elements
        'ADJ', 'DET',                                # Nominal Modifiers
        'VERB', 'AUX',                               # Verbal Elements
        'ADV', 'PART',                               # Adverbials
        'ADP', 'CCONJ', 'SCONJ'                      # Connectives
    ]

    # Convert counts to English-labeled format
    if m1 == True:
        counts_eng_labels = {pred: {gold: i_value for gold, i_value in o_value.items()} for pred, o_value in counts.items()}
        pred_tags = tag_order
    else:
        counts_eng_labels = {pred: {i2l[gold]: i_value for gold, i_value in o_value.items()} for pred, o_value in counts.items()}
        pred_tags = sorted(counts_eng_labels.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    percents_eng_labels = {}
    for k, v in counts_eng_labels.items():
        if sum(v.values()) != 0:
            percents_eng_labels[k] = {i_key: round(i_value / sum(v.values()) * 100, 2) for i_key, i_value in v.items()}

    # Create and sort DataFrame
    # df = pd.DataFrame.from_dict(percents_eng_labels, orient='index')
    # Create DataFrame with predicted as rows and gold as columns
    df = pd.DataFrame.from_dict(percents_eng_labels, orient='index')

    # Reindex columns (gold labels)
    df = df.reindex(columns=tag_order, fill_value=0)

    # Reindex rows (predicted tags)
    df = df.reindex(index=pred_tags, fill_value=0)
    df = df.T
    # df = df.sort_index(axis=0).sort_index(axis=1).T

    # Use a subtle academic-style color palette
    blend_option = "blend:#fff,#FFD666,#D5D06F,#ABC978,#81C281,#56BB8A"
    custom_palette = sns.color_palette(blend_option, as_cmap=True)

    # Set serif font (e.g., Times New Roman or Computer Modern)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    # Initialize figure
    plt.figure(figsize=(10, 8))

    # Plot heatmap with clean style


    # Set axis labels and title
    if m1 == True:
        m1_accuracy_value = m1_accuracy(counts)
        hm = sns.heatmap(
        df,
        cmap=custom_palette,
        annot = True,
        fmt='.1f',
        cbar_kws={'label': '\nPercentage\n', 'shrink': 0.9}
    )
        title = f"\n{experiment_name} Confusion Matrix:\nM-1 and Gold Labels\n\nAverage M1 Accuracy: {m1_accuracy_value:.2f}%"
        file_name = "M1 and Gold Labels"

    else:
        hm = sns.heatmap(
        df,
        cmap=custom_palette,
        cbar_kws={'label': '\nPercentage\n', 'shrink': 0.9} #,
        #norm=LogNorm()
    )
        title = f"\n{experiment_name} Confusion Matrix:\nLatent Tags and Gold Labels"
        file_name = "Latent Tags and Gold Labels"

    hm.set_title(title, pad=20)
    hm.set_xlabel('\nPredicted Labels\n', labelpad=15)
    hm.set_ylabel('\nGold Labels\n', labelpad=15)

    # Improve layout
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.88)  # Ensure there's room for the title

    plt.savefig(f"content/{experiment_name}/{file_name}.png")

    plt.show()

def top_gold_words(config, gold_word_to_tag, index2word, index2label, pred_to_m1, experiment_name):
    data = []

    for tag, word_dict in gold_word_to_tag.items():
        m1_tag = index2label[pred_to_m1[tag]]

        word_dict_sorted = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
        top_word_indices = [k for i, (k, v) in enumerate(word_dict_sorted.items()) if i < 10]
        top_words = [index2word[k] for k in top_word_indices]

        # Create a row with tag and top words
        row = [tag+1] + [m1_tag] + top_words
        data.append(row)

    # Create DataFrame
    columns = ['tag'] + ["m1 tag"] + [f'word_{i+1}' for i in range(10)]
    df = pd.DataFrame(data, columns=columns)
    print(df)

    # Save to CSV
    df.to_csv(f"{experiment_name}/top_words_{datetime.date.today()}.csv", index=False)

    return df