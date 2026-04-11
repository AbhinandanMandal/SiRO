from utils.helperFunctions import append_list_as_row


class TrainingLogger:

    def __init__(self, config):
        self.csv_path = config.save_result_path
        self.best_model_path = config.best_model_path
        self.best_ratio = 0

        # self.plot_dist_path = config.save_plot_dist_path
        # self.plot_learn_path = config.save_plot_learn_path
        # for saving convergence plot
        # self.plot_convergence_path = config.save_convergence_path

        self.train_loss_history = []
        self.test_acc_history = []
        self.useful_exemplar_history = []
        self.max_intra_dist_history = []
        self.intra_dist_history = []
        self.inter_dist_history = []
        self.min_inter_dist_history = []
        self.ratio_history = []

        # writing csv header
        append_list_as_row(
            self.csv_path,
            ["epoch", "train_loss", "test_acc", "useful_exemplar_%",
             "maxdisintra", "disintra", "interCN_catg",
             "interCN_min", "ratio"]
        )

    def log_epoch(self, epoch, train_loss, test_acc, Info, ratio):

        self.train_loss_history.append(train_loss)  # train loss
        self.test_acc_history.append(test_acc)  # test accuracy
        self.useful_exemplar_history.append(Info[0]*100)  # useful exampler %
        self.max_intra_dist_history.append(Info[1])  # maxdisintra
        self.intra_dist_history.append(Info[2])  # disintra
        self.inter_dist_history.append(Info[3])  # interCN_catg
        self.min_inter_dist_history.append(Info[4])  # interCN_min
        self.ratio_history.append(ratio)  # ratio

        # write csv
        """
        Things that i need to save it as .csv file are
        1. epoch
        2. train_loss
        3. test_acc 
        4. max_intra_dist
        5. min_inter_dist
        6. inter_dist_history
        7. intra_dist_history
        8. useful exampler
        9. ratio
        """
        append_list_as_row(
            self.csv_path,
            [epoch, train_loss, test_acc, Info[0]*100,
             Info[1], Info[2], Info[3], Info[4], ratio]
        )

    # def save_plots(self):

    #     """
    #     plot_distance() takes
    #     1. infoEX = useful examplar
    #     2. max_intra = maximum distance within object
    #     3. min_inter = minimum distance between obj i and other obj of the same class
    #     4. ratio = Info[4]/Info[1] = interCN_min / maxdisintra = min_inter / max_intra
    #     5. path = path for storing plot
    #     """
    #     plot_distance(
    #         self.useful_exemplar_history,
    #         self.max_intra_dist_history,
    #         self.min_inter_dist_history,
    #         self.ratio_history,
    #         self.plot_dist_path
    #     )

    #     # plot_infoex() takes infoEX and path for storing
    #     plot_infoex(
    #         self.useful_exemplar_history,
    #         self.plot_learn_path
    #     )

    #     # plot_ConvergenceCurve() takes
    #     # train_loss_history, test_acc_history, path for stroing
    #     plot_ConvergenceCurve(
    #         self.train_loss_history,
    #         self.test_acc_history,
    #         self.plot_convergence_path
    #     )

