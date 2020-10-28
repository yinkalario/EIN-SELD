class BaseTrainer:
    """ Base trainer class

    """
    def train_step(self, *args, **kwargs):
        """ Perform a training step.

        """
        raise NotImplementedError

    def validate_step(self, *args, **kwargs):
        """ Perform a validation step

        """
        raise NotImplementedError



