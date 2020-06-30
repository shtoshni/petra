from collections import OrderedDict
from torchtext.data import Example, Field, Dataset
import torchtext.data as data
from transformers import BertTokenizer


def get_gpr_mention_ids(tokens, ignore_gpr_tags=True):
    gpr_ids = {'<P>': [], '<A>': [], '<B>': []}
    gpr_tag_ids = []
    entity = None
    for i, token_ in enumerate(tokens):
        token = ''.join(tokens[i:i+3]).upper()

        if token in ['<P>', '<A>', '<B>']:
            gpr_tag_ids += [i, i+1, i+2]

        if entity is not None and token not in ['<P>', '<A>', '<B>']:
            if ignore_gpr_tags:
                gpr_ids[entity].append(i+2-len(gpr_tag_ids))
            else:
                gpr_ids[entity].append(i+2)

        if token in ['<P>', '<A>', '<B>']:
            if entity == token:
                entity = None
            else:
                entity = token

    return (gpr_ids['<P>'][:-2],
            gpr_ids['<A>'][:-2],
            gpr_ids['<B>'][:-2],
            gpr_tag_ids)


class GAPDataset(Dataset):
    """Class for parsing GAP dataset."""

    @staticmethod
    def sort_key(example):
        return len(example.Text)

    def __init__(self, path, field_dict, bert_tokenizer, feedback=False,
                 encoding="utf-8", separator="\t",
                 skip_header=False, max_seq_len=512):

        fields = [('ID', Field(sequential=False, use_vocab=False)),
                  ('Text', field_dict['text']),
                  ('P_ids', field_dict['text']),
                  ('A_coref', field_dict['bool']),
                  ('A_ids', field_dict['text']),
                  ('B_coref', field_dict['bool']),
                  ('B_ids', field_dict['text']),
                  ('URL', None)]

        # Parsing fields come from the header line
        parsing_fields = []
        is_header = True

        examples = []

        with open(path, encoding=encoding) as f:
            counter = 0
            for line in f:
                if feedback:
                    counter += 1
                    if counter > 500:
                        break
                line = line.strip()
                if is_header:
                    is_header = False
                    line = line.replace('-', '_')
                    parsing_fields = line.split(separator)
                    continue

                data_dict = OrderedDict()
                for col_type, col_val in \
                        zip(parsing_fields, line.split(separator)):
                    if 'offset' in col_type:
                        data_dict[col_type] = int(col_val)
                    else:
                        data_dict[col_type] = col_val

                entity_tag_dict = {'A': '<A>', 'B': '<B>', 'Pronoun': '<P>'}
                entity_list = []
                for entity, entity_tag in entity_tag_dict.items():
                    loc = data_dict[entity + '_offset']
                    length = len(data_dict[entity])
                    entity_list.append([entity_tag, length, loc])

                entity_list = sorted(entity_list, key=lambda x: x[2],
                                     reverse=True)
                text = data_dict['Text']
                # Add the special tag markers to locate entities
                for (tag, length, loc) in entity_list:
                    text = (text[:loc] + tag + " " + text[loc:loc + length]
                            + " " + tag + text[loc + length:])

                # Add the special tokens
                text = "[CLS] " + text + " [SEP]"
                tokens = bert_tokenizer.tokenize(text)
                if max_seq_len:
                    tokens = tokens[:max_seq_len]

                p_ids, a_ids, b_ids, tag_ids = get_gpr_mention_ids(tokens)
                tag_ids = sorted(tag_ids, reverse=True)
                for tag_id in tag_ids:
                    tokens.pop(tag_id)

                text_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
                data_dict['Text'] = text_ids
                data_dict['ID'] = int(data_dict['ID'].split('-')[1])

                data_dict['P_ids'] = p_ids
                data_dict['A_ids'] = a_ids
                data_dict['B_ids'] = b_ids

                list_of_stuff = []
                for (col_type, _) in fields:
                    list_of_stuff.append(data_dict[col_type])

                if data_dict:
                    examples.append(
                        Example.fromlist(list_of_stuff, fields))

        super(GAPDataset, self).__init__(examples, fields)

    @classmethod
    def iters(cls, path, batch_size=32, batch_first=True,
              feedback=False, train_file='gap-development.tsv',
              dev_file='gap-validation.tsv', test_file='gap-test.tsv'):
        text_field = Field(sequential=True, use_vocab=False,
                           include_lengths=True,
                           batch_first=batch_first, pad_token=0)
        bool_field = Field(sequential=False, use_vocab=False, unk_token=None,
                           preprocessing=(lambda x: x.lower() == 'true'),
                           batch_first=batch_first)

        field_dict = {'text': text_field, 'bool': bool_field}

        bert_tokenizer = cls.load_bert_tokenizer()
        itos = bert_tokenizer.ids_to_tokens

        train, val, test = GAPDataset.splits(
            path=path, train=train_file, validation=dev_file, test=test_file,
            field_dict=field_dict, bert_tokenizer=bert_tokenizer,
            feedback=feedback)

        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size,
            sort_within_batch=True, shuffle=True, repeat=False)

        return (train_iter, val_iter, test_iter, itos)

    @classmethod
    def load_bert_tokenizer(cls):
        """Returns BERT tokenizer."""
        # Both BERT-base and BERT-large use the same vocabulary
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased", do_lower_case=False,
            never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
        return tokenizer
