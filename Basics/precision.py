# Precision = TP / (TP + FP)

# Recall = TP / (TP + FN)

# For a good model precision and recall values should be high

import accuracy

def precision(y_true, y_pred):
    tp = accuracy.true_positive(y_true, y_pred)
    fp = accuracy.false_positive(y_true, y_pred)

    precision = tp / (tp + fp)
    return precision



def recall(y_true, y_pred):
    tp = accuracy.true_positive(y_true, y_pred)
    fn = accuracy.false_negative(y_true, y_pred)

    recall = tp / (tp + fn)
    return recall


if __name__ == "__main__":
    l1 = [0, 1, 1, 1, 0, 0, 0, 1]
    l2 = [0, 1, 0, 1, 0, 1, 0, 0]
    print(precision(l1, l2))
    print(recall(l1, l2))
