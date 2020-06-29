import numpy as np


def get_fscore(labels, preds, threshold=0.5):
    """Returns the F-score given the true labels, scores, and score threshold."""
    preds = np.asarray(preds) > threshold
    labels = np.asarray(labels)

    true_positives = np.sum(preds * labels)

    denom = np.sum(preds)
    precision = true_positives/denom if denom else 0

    denom = np.sum(labels)
    recall = true_positives/denom if denom else 0

    denom = precision + recall
    f_score = 2 * precision * recall/denom if denom else 0

    return f_score


def find_threshold(all_labels, all_preds):
    """Finds the threshold that maximizes the F-score."""
    max_fscore = 0
    threshold = 0.0

    threshold_range = np.arange(0.0, 1.0, 0.01)
    for cur_threshold in threshold_range:
        cur_fscore = get_fscore(all_labels, all_preds,
                                threshold=cur_threshold)
        if cur_fscore > max_fscore:
            max_fscore = cur_fscore
            threshold = cur_threshold

    return max_fscore, threshold
