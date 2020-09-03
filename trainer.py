from average_meter import AverageMeter
from progress_meter import ProgressMeter
import time
import torch
from metrics import accuracy
from util import save_checkpoint


class Trainer(object):
    def __init__(self, train_loader, val_loader, model, criterion, optimizer, optimizer_adjuster, args, train_sampler=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.optimizer_adjuster = optimizer_adjuster
        self.train_sampler = train_sampler

    def train_model(self, start_epoch, end_epoch):
        global best_acc1
        for epoch in range(start_epoch, end_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.optimizer_adjuster.adjust_optimizer(self.optimizer, epoch)

            # train for one epoch
            self.train(epoch)

            # evaluate on validation set
            acc1 = self.validate()

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                        and self.args.rank % self.ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)

    def train(self, epoch):
        """
        Train for a single epoch
        :param train_loader: an enumerable of the train dataset
        :param model: Model to train
        :param criterion: The loss function criterion
        :param optimizer: (Such as SGD, SGD momentum, Adam etc.)
        :param epoch: the current epoch (for logging purposes)
        :param args: the args to run with
        :return: Prints the progress
        """

        batch_time, losses, top1, top5, data_time, progress = get_epoch_meters(self.train_loader, epoch)

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (images, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            self.per_batch(images, target, losses, top1, top5, self.args.gpu)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                progress.display(i)

    def validate(self):
        """
        Validate the model
        :param val_loader: An enumerable of the validation dataset
        :param model: The model we validate
        :param criterion: The loss function (criteria)
        :param args: General run args
        :return: Prints the top 1st and 5th accuracy, and returns the top 1st average accuracy
        """
        batch_time, losses, top1, top5, data_time, progress = get_epoch_meters(self.val_loader)

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.val_loader):
                data_time.update(time.time() - end)
                self.per_batch(images, target, losses, top1, top5, self.args.gpu)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        return top1.avg

    def per_batch(self, images, target, losses_meter, top1_meter, top5_meter, gpu=None):
        """
        Executes the model over a batch of data, and updates the meters. If given an optimizer, it will update the model using backprop
        :param model: Our model
        :param criterion: The loss function
        :param images: batch of images
        :param target: correct labels of images
        :param losses_meter: meter for loss
        :param top1_meter: meter for top1 accuracy
        :param top5_meter: meter for top5 accuracy
        :param gpu: The specific GPU we wish to use for loading the images. If None then we won't use the gpu for this purpose.
        Defaults to None.
        :param optimizer: The optimizer we use to train the model. If None then we won't train the model on the data
        Defaults to None
        :return: None
        """
        if gpu is not None:
            images = images.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = self.model(images)
        loss = self.criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1[0], images.size(0))
        top5_meter.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def get_epoch_meters(data_loader, epoch=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    data_time = AverageMeter('Data', ':6.3f')
    prefix = 'Test: '
    if epoch is not None:
        prefix = 'Epoch: [{}]'.format(epoch)
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=prefix)

    return batch_time, losses, top1, top5, data_time, progress
