
class LearningRateFactorDecreaser(object):
    def __init__(self, factor, epochs_interval):
        self.factor = factor
        self.epochs_interval = epochs_interval

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if (epoch % self.epochs_interval) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
