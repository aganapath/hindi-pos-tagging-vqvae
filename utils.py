import regex
from sklearn.model_selection import train_test_split

def read_conllu_file(file_path: str):
    """Read CoNLL-U format files for labeled data"""
    sentences = []
    unique_labels = set()

    with open(file_path, "r", encoding="UTF-8") as in_f:
        current_sentence = []
        for line in in_f:
            line = line.strip()
            if line.startswith("#") or line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            parts = line.split("\t")
            idx = parts[0]

            if "." in idx or "-" in idx or len(parts) < 4:
                continue

            word, tag = parts[1], parts[3]
            word = regex.sub(r"[^\p{Devanagari}+]", "", word)
            if word != "":
                unique_labels.add(tag)
                current_sentence.append((word, tag))

    return sentences, unique_labels

def read_parquet_file(file_path: str, n: int, random_state: int, train_size: int, test_size: int):
    """Read parquet files for unlabeled data"""
    df = pd.read_parquet(file_path)
    df_sampled = df.sample(n=n, random_state=random_state).to_numpy()
    df_train, df_dev = train_test_split(df_sampled, train_size=train_size, test_size=test_size, random_state=random_state)

    def clean_parquet_data(array: np.array):
        sentences = []
        for str_data in array:
            sentence: str = str_data[0]['hi']
            cleaned_sentence: list = regex.findall(r"\p{Devanagari}+", sentence)
            sentences.append(cleaned_sentence)
        return sentences

    train_sents = clean_parquet_data(df_train)
    dev_sents = clean_parquet_data(df_dev)

    return train_sents, dev_sents

def item_indexer(list_of_items, labels=False):
    item2index = {item: idx+1 for idx, item in enumerate(dict.fromkeys(sorted(list_of_items)))}

    if not labels:
        item2index["UNK"] = len(item2index)
        item2index["PAD"] = 0

    index2item = {idx: item for item, idx in item2index.items()}

    return item2index, index2item