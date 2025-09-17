def pred_to_gold(gold_predicted_joined, config, label2index):
    # count_dict = {i: {i: 0 for i in range(config.num_tag)} for i in range(len(label2index))}
    count_dict = {i: {i+1: 0 for i in range(len(label2index))} for i in range(config.num_tag)}

    for seq in gold_predicted_joined:
        for tup_tag in seq:
            gold_tag = int(tup_tag[0])
            pred_tag = int(tup_tag[1])

            if gold_tag != -100:
                # count_dict[gold_tag][pred_t ag] = count_dict[gold_tag].get(pred_tag, 0) + 1
                count_dict[pred_tag][gold_tag] = count_dict[pred_tag].get(gold_tag, 0) + 1

    return count_dict

def many_to_one(gold_predicted_joined, config, label2index, index2label):
    count_dict = pred_to_gold(gold_predicted_joined, config, label2index)

    m1_dict = {i+1: [] for i in range(len(label2index))}
    pred_to_m1 = {i: 0 for i in range(config.num_tag)}

    for pred, gold_counts in count_dict.items():
        # gets the gold tag most associated with each predicted tag
        top_gold_tag = max(gold_counts, key=gold_counts.get)
        m1_dict[top_gold_tag].append(gold_counts)
        pred_to_m1[pred] = top_gold_tag

    # investigating m1_dict
    m1_freq_dict = {}
    for m1_tag, dict_list in m1_dict.items():
        tag = index2label[m1_tag]
        freq_m1 = len(dict_list)
        m1_freq_dict[tag] = freq_m1

    sorted_m1_freq = dict(sorted(m1_freq_dict.items(), key=lambda item: item[1], reverse=True))
    top_tags = {k: v for i, (k, v) in enumerate(sorted_m1_freq.items()) if i < 5}
    print(f"the top 5 m1 tags are: {top_tags}")

    # aggregate all of the individual gold tag dicts
    m1_totals = {}
    for m1_tag, dict_list in m1_dict.items():
        totals = {}
        for d in dict_list:
            for k, v in d.items():
                if k not in totals:
                    totals[index2label[k]] = v
                else:
                    totals[index2label[k]] += v

        m1_totals[index2label[m1_tag]] = totals

    return m1_totals, pred_to_m1

def m1_accuracy(m1_totals):
    correct = 0
    total = 0
    for k, count_dict in m1_totals.items():
        if k in count_dict.keys():
            k_correct = count_dict[k]
            correct += k_correct

        k_total = sum(count_dict.values())
        total += k_total

    accuracy = (correct / total) * 100
    print(f"Overall M1 accuracy is {accuracy:.2f}%")

    return accuracy