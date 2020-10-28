import methods.utils.SELD_evaluation_metrics_2019 as SELDMetrics2019
from methods.utils.SELD_evaluation_metrics_2020 import \
    SELDMetrics as SELDMetrics2020
from methods.utils.SELD_evaluation_metrics_2020 import early_stopping_metric


class Metrics(object):
    """Metrics for evaluation

    """
    def __init__(self, dataset):

        self.metrics = []
        self.names = ['ER20', 'F20', 'LE20', 'LR20', 'seld20', 'ER19', 'F19', 'LE19', 'LR19', 'seld19']

        self.num_classes = len(dataset.label_set)
        self.doa_threshold = 20 # in deg
        self.num_frames_1s = int(1 / dataset.label_resolution)

    def calculate(self, pred_dict, gt_dict):

        # ER20: error rate, F20: F1-score, LE20: Location error, LR20: Location recall
        ER_19, F_19 = SELDMetrics2019.compute_sed_scores(pred_dict['dcase2019_sed'], gt_dict['dcase2019_sed'], \
            self.num_frames_1s)
        LE_19, LR_19, _, _, _, _ = SELDMetrics2019.compute_doa_scores_regr( \
            pred_dict['dcase2019_doa'], gt_dict['dcase2019_doa'], pred_dict['dcase2019_sed'], gt_dict['dcase2019_sed'])
        seld_score_19 = SELDMetrics2019.early_stopping_metric([ER_19, F_19], [LE_19, LR_19])

        dcase2020_metric = SELDMetrics2020(nb_classes=self.num_classes, doa_threshold=self.doa_threshold)
        dcase2020_metric.update_seld_scores(pred_dict['dcase2020'], gt_dict['dcase2020'])
        ER_20, F_20, LE_20, LR_20 = dcase2020_metric.compute_seld_scores()
        seld_score_20 = early_stopping_metric([ER_20, F_20], [LE_20, LR_20])

        metrics_scores = {
            'ER20': ER_20,
            'F20': F_20,
            'LE20': LE_20,
            'LR20': LR_20,
            'seld20': seld_score_20,
            'ER19': ER_19,
            'F19': F_19,
            'LE19': LE_19,
            'LR19': LR_19,
            'seld19': seld_score_19,
        }
        return metrics_scores
