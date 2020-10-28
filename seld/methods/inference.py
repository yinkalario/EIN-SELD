class BaseInferer:
    """ Base inferer class

    """
    def infer(self, *args, **kwargs):
        """ Perform an inference on test data.

        """
        raise NotImplementedError

    def fusion(self, submissions_dir, preds):
        """ Ensamble predictions.

        """
        raise NotImplementedError        

    @staticmethod
    def write_submission(submissions_dir, pred_dict):
        """ Write predicted result to submission csv files
        Args:
            pred_dict: DCASE2020 format dict:
                pred_dict[frame-containing-events] = [[class_index_1, azi_1 in degree, ele_1 in degree], [class_index_2, azi_2 in degree, ele_2 in degree]]
        """
        for key, values in pred_dict.items():
            for value in values:
                with submissions_dir.open('a') as f:
                    f.write('{},{},{},{}\n'.format(key, value[0], value[1], value[2]))



