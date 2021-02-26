# When we have an equal no of positive and negative samples in a binary classification metric,
# we generally use accuracy, precision, recall and f1

# If your model correctly predicts positive class, it is true positive, and if your model accurately predicts
# negative class, it is a true negative.

# If your model incorrectly(or falsely) predicts positive class, it is a false positive. If your model
# incorrectly(or falsely) predicts negative class, it is a false negative.


from sklearn import metrics


def accuracy(y_true, y_pred):
    correct_counter = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1

    return correct_counter / len(y_true)


# we can calculate accuracy using scikit-learn as well

def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp


def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn


def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp


def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn


def accuracy_v2(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    accuracy_score = (tp + tn) / (tp + tn + fp + fn)
    return accuracy_score


if __name__ == "__main__":
    l1 = [0, 1, 1, 1, 0, 0, 0, 1]
    l2 = [0, 1, 0, 1, 0, 1, 0, 0]

    print(metrics.accuracy_score(l1, l2))


    print(accuracy(l1, l2))

    print(accuracy_v2(l1, l2))
