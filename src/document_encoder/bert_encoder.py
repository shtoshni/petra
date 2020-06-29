import torch
import torch.nn as nn
from transformers import BertModel
from pytorch_utils.modules import MLP


class BertEncoder(nn.Module):
    def __init__(self, model_size='base', use_enc_mlp=False,
                 mem_size=300, mlp_size=300, **kwargs):
        super(BertEncoder, self).__init__()

        self.last_layers = 4
        if model_size == 'large':
            self.start_layer_idx = 19
            self.end_layer_idx = 23
        elif model_size == 'base':
            self.start_layer_idx = 9
            self.end_layer_idx = 13

        # Summary Writer
        self.bert = BertModel.from_pretrained('bert-' + model_size + '-cased',
                                              output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        encoder_hidden_size = 4 * bert_hidden_size

        # The BERT output is fed to RNN or MLP
        # In the paper, we use GRU. Experiments with MLP can run into NaN issues.
        self.use_enc_mlp = use_enc_mlp
        if not self.use_enc_mlp:
            self.rnn = nn.GRU(
                input_size=encoder_hidden_size,  # + mem_size + 2,
                hidden_size=mem_size, batch_first=True)
        else:
            self.enc_mlp = MLP(input_size=encoder_hidden_size, hidden_size=mem_size,
                               output_size=mem_size, num_layers=2, bias=True)

    def encode_documents(self, batch_documents, input_mask):
        """
        Encode a batch of documents.
        batch_excerpt: B x L
        input_mask: B x L
        """
        batch_size, max_len = batch_documents.size()

        with torch.no_grad():
            outputs = self.bert(batch_documents, attention_mask=input_mask)  # B x L x E

        encoded_layers = outputs[2]
        encoded_repr = torch.cat(
            encoded_layers[self.start_layer_idx:self.end_layer_idx], dim=-1)

        # Prpject the document encoding to memory size.
        if not self.use_enc_mlp:
            encoded_doc, _ = self.rnn(encoded_repr)
        else:
            encoded_doc = torch.tanh(self.query_mlp(encoded_repr))

        return encoded_doc
