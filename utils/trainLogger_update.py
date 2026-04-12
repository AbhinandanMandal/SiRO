
import os
import pandas as pd


class TrainingLogger:
    def __init__(self, Config):
        self.csv_path = Config.save_result_path
        self.best_model_path = Config.best_model_path
        self.best_ratio = 0

        # columns of .csv file
        # maxdisintra: max_intra_dist_history = maximum distance within object
        # disintra = intra_dist_history = average distance within object
        # interCN_catg = inter_dist_history = avergae distance between object i and other object of the same class
        # interCN_min = min_inter_dist_history = minimum distance between object i and other object of the same class
        # ratio = interCN_min/maxdisintra
        self.columns = [
            "epoch",
            "train_loss",
            "test_acc",
            "useful_exemplar_%",
            "maxdisintra",
            "disintra",
            "interCN_catg",
            "interCN_min",
            "ratio"
        ]

        # loading existing path or creating new path for saving .csv file
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
        else:
            self.df = pd.DataFrame(columns=self.columns)

    # storing and saving the file
    def log_epoch(self, epoch, train_loss, test_acc, Info, ratio):
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "useful_exemplar_%": Info[0]*100,
            "maxdisintra": Info[1],
            "disintra": Info[2],
            "interCN_catg": Info[3],
            "interCN_min": Info[4],
            "ratio": ratio
        }

        self.df.loc[len(self.df)] = row  # append df into .csv file
        self.df.to_csv(self.csv_path, index=False)  # saving the file
