# F1 score is a metric that combines both precision and recall
# Its the harmonic mean of precision and recall

# F1 = 2PR/(P + R)

# F1 = 2TP / (2TP + FP + FN)
from sklearn import metrics

import precision


def f1(y_true, y_pred):
    p = precision.precision(y_true, y_pred)
    r = precision.recall(y_true, y_pred)

    score = 2 * p * r / (p + r)
    return score


if __name__ == "__main__":
    y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
              1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
              1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    print(f1(y_true, y_pred))

    print(metrics.f1_score(y_true, y_pred))
