from gap_utils.gap import GAPDataset
from itertools import product


def get_iter(data_dir, batch_size=32, model='base'):
    # print ("Max story size:", memory_size)
    train_iter, valid_iter, test_iter, itos = GAPDataset.iters(
        path=data_dir, batch_size=batch_size,
        model=model)

    return (train_iter, valid_iter, test_iter, itos)


def gen_ent_coref(ent_ids):
    ent_coref = {}
    id1 = ent_ids[0]
    for id2 in ent_ids[1:]:
        if id2:
            ent_coref[frozenset([id1, id2])] = ('Same', 1)
            # break
    return ent_coref


def gen_ent2ent_coref(ent1_ids, ent2_ids, ent, ent_coref):
    ent2ent_coref = {}
    for id1, id2 in product(ent1_ids, ent2_ids):
        if id1 and id2:
            # Check that both of the ids are not MASK
            ent2ent_coref[frozenset([id1, id2])] = (ent, ent_coref)
            # break
    return ent2ent_coref


def get_all_coref_pairs(data, validation=False):
    a_ids, a_len = data.A_ids
    batch_size = a_ids.shape[0]
    text, text_length = data.Text

    batch_coref_pairs = []
    for i in range(batch_size):
        batch_coref_pairs.append({})

        offset_len_pairs = {}
        offset_len_pairs['P'] = data.P_ids[0][i]
        offset_len_pairs['A'] = data.A_ids[0][i]
        offset_len_pairs['B'] = data.B_ids[0][i]

        if not validation:
            # Coref pairs for multi-token mention spans
            for ent, ent_info in offset_len_pairs.items():
                batch_coref_pairs[i].update(gen_ent_coref(ent_info))

        # Coref pairs for multi-token mention spans
        batch_coref_pairs[i].update(gen_ent2ent_coref(
            offset_len_pairs['P'], offset_len_pairs['A'],
            'A', data.A_coref[i]))

        batch_coref_pairs[i].update(gen_ent2ent_coref(
            offset_len_pairs['P'], offset_len_pairs['B'],
            'B', data.B_coref[i]))

        # In GAP, mentions A & B correspond to different entities
        batch_coref_pairs[i].update(gen_ent2ent_coref(
            offset_len_pairs['A'], offset_len_pairs['B'],
            'AB', 0))

    return batch_coref_pairs


def bert_tokens_to_str(text_ids, token_ids, itos):
    """Combine BERT tokens."""
    batch_size, _ = token_ids.shape
    output_str_list = []
    for i in range(batch_size):
        output_str = ""
        token_list = token_ids[i, :].tolist()
        for token_id in token_list:
            if token_id:
                # Not a PAD symbol
                token = itos[text_ids[i, token_id].item()]
                if token[:2] == "##":
                    output_str += token[2:]
                else:
                    output_str += " " + token
        output_str_list.append(output_str)
    return output_str_list


if __name__ == '__main__':
    DATA_DIR = "../data/"
    _, valid_iter, _, itos = get_iter(DATA_DIR, batch_size=2)
    print(len(valid_iter.data()))
    for batch in valid_iter:
        print(batch.ID)
        print(batch.A_coref)
        print(batch.B_coref)
        break
