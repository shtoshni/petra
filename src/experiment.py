import sys
from os import path
import os

import time
import logging
import torch
import json

import pytorch_utils.utils as utils
from controller import Controller
from gap_utils.gap import GAPDataset
from gap_utils.data_utils import bert_tokens_to_str

from gap_utils.gap_scorer import run_scorer
from gap_utils.gap_utils import get_fscore, find_threshold

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class Experiment:
    def __init__(self, data_dir=None, model_dir=None, best_model_dir=None,
                 # Training params
                 max_epochs=75, max_num_stuck_epochs=15, eval=False, feedback=False,
                 batch_size=32, seed=0, init_lr=1e-3, ent_loss=0.1,
                 # Slurm params
                 slurm_id=None,
                 # Other params
                 **kwargs):

        # Set training params
        self.max_epochs = max_epochs
        self.max_num_stuck_epochs = max_num_stuck_epochs
        self.feedback = feedback
        self.slurm_id = slurm_id
        self.ent_loss = ent_loss

        # Prepare data info
        self.train_iter, self.valid_iter, self.test_iter, self.itos \
            = GAPDataset.iters(path=data_dir, batch_size=batch_size, feedback=feedback)

        # Get model paths
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_path = path.join(model_dir, 'model.pth')
        self.best_model_path = path.join(best_model_dir, 'model.pth')

        # Initialize model and training metadata
        self.model = Controller(**kwargs)
        self.model = self.model.cuda()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=init_lr, weight_decay=0)
        self.optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5,
            min_lr=0.1 * init_lr, verbose=True)

        self.train_info = {}
        self.train_info['epoch'] = 0
        self.train_info['val_perf'] = 0.0
        self.train_info['threshold'] = 0.0
        self.train_info['global_steps'] = 0
        self.train_info['num_stuck_epochs'] = 0

        if not eval:
            self.initialize_setup(seed)
            self.train()
        # Finally evaluate model
        self.final_eval(model_dir)

    def initialize_setup(self, seed):
        """Initialize model and training info."""
        if not path.exists(self.model_path):
            torch.manual_seed(seed)
        else:
            logging.info('Loading previous model: %s' % (self.model_path))
            # Load model
            self.load_model(self.model_path)

        utils.print_model_info(self.model)

    def train(self):
        """Train model"""
        model = self.model
        epochs_done = self.train_info['epoch']
        optimizer = self.optimizer
        scheduler = self.optim_scheduler

        if self.train_info['num_stuck_epochs'] >= self.max_num_stuck_epochs:
            return

        for epoch in range(epochs_done, self.max_epochs):
            if self.train_info['num_stuck_epochs'] >= self.max_num_stuck_epochs:
                # Exit training if model hasn't improved in a while
                return

            print("\n\nStart Epoch %d" % (epoch + 1))
            start_time = time.time()
            model.train()

            if (epoch + 1) % 10 == 0:
                # Reduce Gumbel-Softmax Temperature
                self.model.memory_net.gumbel_temperature /= 2.0
            for train_batch in self.train_iter:
                self.train_info['global_steps'] += 1
                loss = model(train_batch)
                total_loss = loss['coref']
                total_loss += loss['ent'] * self.ent_loss

                # Gradient updates
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Update epochs done
            self.train_info['epoch'] = epoch + 1
            # Validation performance
            val_loss, fscore, threshold, _ = self.eval_model()
            scheduler.step(fscore)

            # Update model if validation performance improves
            if fscore > self.train_info['val_perf']:
                self.train_info['val_perf'] = fscore
                self.train_info['threshold'] = threshold
                logging.info('Saving best model')
                self.save_model(self.best_model_path)

                # Reset num_stuck_epochs
                self.train_info['num_stuck_epochs'] = 0
            else:
                self.train_info['num_stuck_epochs'] += 1

            # Save model
            self.save_model(self.model_path)

            # Get elapsed time
            elapsed_time = time.time() - start_time
            logging.info("Epoch: %d, Time: %.2f, F-score: %.3f (Max: %.3f)"
                         % (epoch + 1, elapsed_time, fscore,
                            self.train_info['val_perf']))
            sys.stdout.flush()

    def eval_model(self, split='valid', threshold=None, final_eval=False):
        """Eval model"""
        # Set the random seed to get consistent results
        model = self.model
        model.eval()

        id_prefix, data_iter = None, None
        if split == 'valid':
            data_iter = self.valid_iter
            id_prefix = 'validation'
        elif split == 'train':
            data_iter = self.train_iter
            id_prefix = 'development'
        else:
            data_iter = self.test_iter
            id_prefix = 'test'

        with torch.no_grad():
            all_vars = {}
            total_loss = 0.0
            total_weight = 0.0
            # Output file to write the outputs
            if final_eval:
                suffix = (id_prefix + '-final')
            else:
                suffix = id_prefix + '-last'
            output_file = path.join(self.model_dir, 'pred-' + suffix + '.tsv')

            json_dump_file = path.join(
                self.model_dir, 'json-' + suffix + '.js')

            agg_results = []
            for j, data_batch in enumerate(data_iter):
                inst_id = data_batch.ID.tolist()
                text, text_length = data_batch.Text
                batch_loss, batch_weight, outputs, preds, y = model(data_batch)
                total_loss += batch_loss.item()
                total_weight += batch_weight.item()
                coref, overwrite, ent, usage =\
                    (outputs['coref'], outputs['overwrite'], outputs['ent'], outputs['usage'])

                batch_size = len(inst_id)
                a_ents = bert_tokens_to_str(text_ids=text,
                                            token_ids=data_batch.A_ids[0],
                                            itos=self.itos)
                b_ents = bert_tokens_to_str(
                    text_ids=text, token_ids=data_batch.B_ids[0],
                    itos=self.itos)

                a_coref = data_batch.A_coref.tolist()
                b_coref = data_batch.B_coref.tolist()

                for i in range(batch_size):
                    agg_results.append([str(inst_id[i]), y[2*i], preds[2*i]])
                    agg_results.append([str(inst_id[i]), y[2*i + 1],
                                        preds[2*i + 1]])
                    all_vars[str(inst_id[i])] = {
                        'text': [self.itos[text_id] for text_id
                                 in text[i][:text_length[i]].tolist()],
                        'ent_names': [a_ents[i], b_ents[i]],
                        'g_coref': [str(a_coref[i]), str(b_coref[i])],
                        'coref': [coref[j][i].tolist() for j in range(text_length[i])],
                        'overwrite': [overwrite[j][i].tolist() for j in range(text_length[i])],
                        'usage': [usage[j][i].tolist() for j in range(text_length[i])],
                        'ent': ['{:.3f}'.format(ent[j][i].item()) for j in range(text_length[i])],
                    }

        all_ids, all_labels, all_preds = zip(*agg_results)
        avg_loss = total_loss/total_weight
        if threshold:
            max_fscore = get_fscore(
                all_labels=all_labels, all_preds=all_preds,
                threshold=threshold)
        else:
            max_fscore, threshold = find_threshold(all_labels, all_preds)

            logging.info("Max F-score: %.3f, Threshold: %.2f" %
                         (max_fscore, threshold))

        with open(output_file, 'w') as f:
            str_preds = ['TRUE' if score >= threshold else 'FALSE'
                         for score in all_preds]
            for i in range(0, len(all_ids), 2):
                all_vars[all_ids[i]]['preds'] = [
                    (str_preds[i] == 'TRUE'), (str_preds[i+1] == 'TRUE')]
                all_vars[all_ids[i]]['scores'] = [all_preds[i], all_preds[i+1]]
                errors = int((str_preds[i] == 'TRUE') != all_labels[i])
                errors += int((str_preds[i+1] == 'TRUE') != all_labels[i+1])
                all_vars[all_ids[i]]['errors'] = errors
                f.write(id_prefix + "-" + str(all_ids[i]) + "\t"
                        + str_preds[i] + "\t"
                        + str_preds[i + 1] + "\n")

            all_vars['Threshold'] = threshold.item()

        if split == 'valid' or split == 'test':
            # JSON dump is useful to analyze the model
            with open(json_dump_file, "w") as dump_f:
                data_str = json.dumps(all_vars, indent=2)
                # Make it a JS file which we use to visualize the logs
                data_str = "var data=" + data_str
                dump_f.write(data_str)

        return avg_loss, max_fscore, threshold, output_file

    def final_eval(self, model_dir):
        """Evaluate the model on train, dev, and test"""
        # Test performance  - Load best model
        self.load_model(self.best_model_path)
        logging.info("Loaded best model after epoch: %d" %
                     self.train_info['epoch'])
        logging.info("Threshold: %.2f" % self.train_info['threshold'])
        threshold = self.train_info['threshold']

        perf_file = path.join(self.model_dir, "perf.txt")
        if self.slurm_id:
            parent_dir = path.dirname(path.normpath(self.model_dir))
            perf_dir = path.join(parent_dir, "perf")
            if not path.exists(perf_dir):
                os.makedirs(perf_dir)
            perf_file = path.join(perf_dir, self.slurm_id + ".txt")

        with open(perf_file, 'w') as f:
            f.write(self.best_model_path + '\n')
            for split in ['Train', 'Valid', 'Test']:
                logging.info('\n')
                logging.info('%s' % split)
                split_loss, split_f1, _, split_file = self.eval_model(
                    split.lower(), threshold=threshold, final_eval=True)
                logging.info('Calculated F1: %.3f' % split_f1)
                logging.info('Output at: %s\n' % split_file)
                if split == 'Train':
                    gold_file = 'gap-development.tsv'
                elif split == 'Valid':
                    gold_file = 'gap-validation.tsv'
                elif split == 'Test':
                    gold_file = 'gap-test.tsv'

                f.write("%s\t%.4f\n" % (split, split_f1))

            logging.info("Final performance summary at %s" % perf_file)
            if not self.feedback:
                scoreboard = run_scorer(path.join(self.data_dir, gold_file),
                                        split_file)
                logging.info(scoreboard)

        sys.stdout.flush()

    def load_model(self, location):
        checkpoint = torch.load(location)
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        self.optim_scheduler.load_state_dict(
            checkpoint['scheduler_state_dict'])
        self.train_info = checkpoint['train_info']
        torch.set_rng_state(checkpoint['rng_state'])

    def save_model(self, location):
        """Save model"""
        save_dict = {}
        save_dict.update({
            'train_info': self.train_info,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optim_scheduler.state_dict(),
            'rng_state': torch.get_rng_state(),
        })
        torch.save(save_dict, location)
        logging.info("Model saved at: %s" % (location))
