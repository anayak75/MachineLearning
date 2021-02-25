import pandas as pd
from sklearn import model_selection
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv('/Users/alok/Documents/data/winequality-red.csv', sep=';')
    quality_mapping = {
        3: 0,
        4: 1,
        5: 2,
        6: 3,
        7: 4,
        8: 5
    }

    df.loc[:, "quality"] = df.quality.map(quality_mapping)

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    df.to_csv("/Users/alok/Documents/data/train_folds.csv", index=False)
