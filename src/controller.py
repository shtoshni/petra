import torch
import torch.nn as nn

from collections import defaultdict
from document_encoder.bert_encoder import BertEncoder
from pytorch_utils.utils import get_sequence_mask
from gap_utils.data_utils import get_all_coref_pairs
from memory.working_memory import WorkingMemory

EPS = 1e-8


class Controller(nn.Module):
    def __init__(self, cumm='sum', **kwargs):
        super(Controller, self).__init__()
        self.cumm = cumm
        self.doc_encoder = BertEncoder(**kwargs)
        self.memory_net = WorkingMemory(**kwargs)
        self.num_cells = self.memory_net.num_cells

    def get_ent_loss(self, batch_data, input_mask, ent_list):
        """Loss for predicting entities outside the labeled spans."""
        ent_pred = torch.transpose(torch.stack(ent_list), 0, 1)  # B x T
        ent_pred.squeeze_(dim=2).cuda()

        ent_weight = torch.ones_like(ent_pred).cuda()
        ent_weight = ent_weight * input_mask

        # Mask out entity predictions for labeled spans
        for ids, ids_len in [batch_data.P_ids, batch_data.A_ids, batch_data.B_ids]:
            batch_size, ids_time = ids.size()
            for i in range(batch_size):
                for j in range(ids_time):
                    if ids[i, j]:
                        ent_weight[i, ids[i, j]] = 0
        ent_weight[:, 0] = 1

        other_ents = torch.sum(ent_pred * ent_weight)/torch.sum(ent_weight)
        return other_ents

    def predict_output(self, prob, coref_pairs_labels):
        best_prob_list = []
        label_list = []
        output = []

        batch_size = prob.shape[0]
        for i in range(batch_size):
            best_prob_list.append(defaultdict(float))
            label_list.append({})
            for (ind1, ind2), (ent, label) in coref_pairs_labels[i].items():
                label_list[i][ent] = label
                t1, t2 = ind1, ind2
                if t1 > t2:
                    t1, t2 = t2, t1
                coref_prob = prob[i, t1, t2]

                output.append((coref_prob, label))
                if coref_prob > best_prob_list[i][ent]:
                    best_prob_list[i][ent] = coref_prob.item()
        final_pred_list = []
        final_label_list = []

        for best_prob_dict, label_dict in zip(best_prob_list, label_list):
            final_pred_list.append(best_prob_dict['A'])
            final_label_list.append(label_dict['A'].item())

            final_pred_list.append(best_prob_dict['B'])
            final_label_list.append(label_dict['B'].item())

        return final_pred_list, final_label_list

    def predict_pairwise_prob(self, outputs):
        """Predict pairwise coref probability based on coref & overwrite probs.
        Not that our loss calculations don't require calculating all pair probs.
        But for short documents, all pair implementation is easier & faster.
        """
        coref_prob_list = outputs['coref']
        overwrite_prob_list = outputs['overwrite']

        batch_size, num_cells = coref_prob_list[0].size()
        time_steps = len(coref_prob_list)

        # First time idx is t1 & second idx is t2
        prob_tens = torch.zeros(batch_size, time_steps,
                                time_steps, num_cells).cuda()

        upper_tri_mask = torch.triu(
            torch.ones(batch_size, time_steps, time_steps), diagonal=1).cuda()
        upper_tri_mask.unsqueeze_(dim=3)

        for t in range(time_steps):
            # t1
            sum_slice = torch.log(
                coref_prob_list[t] + overwrite_prob_list[t])  # B x M
            sum_slice = torch.unsqueeze(sum_slice, dim=1).expand(
                -1, time_steps, -1)  # B x T x M
            prob_tens[:, t, :, :] += sum_slice

            # t2
            coref_slice = torch.log(coref_prob_list[t])  # B x M
            coref_slice = torch.unsqueeze(coref_slice, dim=1).expand(
                -1, time_steps, -1)  # B x T x M
            prob_tens[:, :, t, :] += coref_slice

        prob_tens = prob_tens * upper_tri_mask
        # Now let's calculate the sum of log of (1 - overwrite_prob[t][i])
        overwrite_tens = torch.stack(overwrite_prob_list, dim=1)  # B x T x M
        no_overwrite_tens = (1 - overwrite_tens) * (1 - EPS) + EPS
        no_overwrite_tens = torch.log(no_overwrite_tens)  # B x T x M

        # Maintains the summation of first t no_overwrite time steps.
        sum_no_overwrite_tens = torch.zeros_like(no_overwrite_tens).cuda()
        for t in range(time_steps):
            if t == 0:
                sum_no_overwrite_tens[:, 0, :] = no_overwrite_tens[:, 0, :]
            else:
                sum_no_overwrite_tens[:, t, :] = (
                    sum_no_overwrite_tens[:, t-1, :]
                    + no_overwrite_tens[:, t, :])

        t1_sum_tens = torch.repeat_interleave(
            sum_no_overwrite_tens, time_steps, dim=1)
        t2_sum_tens = sum_no_overwrite_tens.repeat(1, time_steps, 1)

        # Only the upper triangular entries are used, so the following
        # expression works
        diff_overwrite_tens = t2_sum_tens - t1_sum_tens
        diff_overwrite_tens = diff_overwrite_tens.reshape_as(prob_tens)
        # It's important to mask out the lower-triangular entries because
        # some of them are inverse of prob product and hence, not probabilities
        # by themselves.
        diff_overwrite_tens = diff_overwrite_tens * upper_tri_mask
        assert(torch.max(diff_overwrite_tens) <= 0)
        # Now add this term to the other terms
        prob_tens = prob_tens + diff_overwrite_tens  # B x T x T x M
        if self.cumm == 'sum':
            prob = torch.logsumexp(prob_tens, dim=3)
        else:
            prob, _ = torch.max(prob_tens, dim=3)
        prob = prob * torch.squeeze(upper_tri_mask, dim=-1)

        return prob

    def forward(self, batch_data):
        """
        Encode a batch of excerpts.
        """
        text, text_length = batch_data.Text
        text, text_length = text.cuda(), text_length.cuda()

        attn_mask = get_sequence_mask(text_length).cuda().float()
        encoded_doc = self.doc_encoder.encode_documents(text, attn_mask)

        batch_size = encoded_doc.shape[0]
        hidden_state_list = torch.unbind(encoded_doc, dim=1)

        # Now change the attn mask to input mask where we zero out [CLS] and [SEP]
        input_mask = attn_mask
        # [CLS] and [SEP] shouldn't correspond to entities
        input_mask[:, 0] = 0  # [CLS]
        input_mask[torch.arange(text.shape[0]), text_length - 1] = 0  # [SEP]

        input_mask_list = torch.unbind(input_mask, dim=1)
        outputs = self.memory_net(hidden_state_list, input_mask_list)
        prob = self.predict_pairwise_prob(outputs)

        # GT pair labels
        coref_pairs_labels = get_all_coref_pairs(batch_data, validation=not self.training)

        if self.training:
            # Weight of our loss
            weight = torch.zeros_like(prob).cuda()
            y = torch.zeros_like(weight)
            for i in range(batch_size):
                for (ind1, ind2), (ent, label) in coref_pairs_labels[i].items():
                    t1, t2 = ind1, ind2
                    if t1 > t2:
                        t1, t2 = t2, t1
                    # Assign different weights to different labels
                    if ent == 'Same':
                        weight[i, t1, t2] = 1.0  # Coref within same span
                    else:
                        weight[i, t1, t2] = 5.0  # Coref across spans
                        if (not label):
                            weight[i, t1, t2] = 50  # Not coreferent

                    y[i, t1, t2] = label

            coref_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prob, y, weight=weight, reduction='sum')
            total_weight = torch.sum(weight)

            loss = {}
            loss['coref'] = coref_loss/total_weight
            loss['ent'] = self.get_ent_loss(batch_data, input_mask, outputs['ent'])
            return loss
        else:
            prob = torch.exp(prob)
            final_pred_list, final_label_list = self.predict_output(
                prob, coref_pairs_labels)
            return (outputs, final_pred_list, final_label_list)
