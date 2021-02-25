'''
python train_your_dataset.py --num-epochs 40  --num-gpus 1 -j 8 --batch-size 32 --wd 0.0001  --lr 0.0001  --lr-decay-epoch 10,20,30 --model resnet18_v2
'''

import matplotlib
matplotlib.use('Agg')
import argparse, time, logging, os
import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
import gluoncv as gcv
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--dataset', type=str, default='/media/cj1/data/minc-2500',
                        help='training and validation pictures to use.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--classes',type=int, default=23,
                        help='number of dataset calsses.')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='resnet18_v2',
                        help='model to use. default is resnet.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=40,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='10,20,30',
                        help='epochs at which learning rate decays. default is 10,20,30.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--save-plot-dir', type=str, default='.',
                        help='the path to save the history plot')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    batch_size = opt.batch_size
    train_path = os.path.join(opt.dataset, 'train')
    val_path = os.path.join(opt.dataset, 'val')
    test_path = os.path.join(opt.dataset, 'test')
    classes = opt.classes
    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]
    model_name = opt.model
    finetune_net = get_model(model_name, pretrained=True)
    if opt.resume_from:
        finetune_net.load_parameters(opt.resume_from, ctx = context)

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0

    plot_path = opt.save_plot_dir

    logging.basicConfig(level=logging.INFO)
    logging.info(opt)

    jitter_param = 0.4
    lighting_param = 0.1
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    def test(finetune_net, ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [finetune_net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        finetune_net.output.initialize(mx.init.Xavier(), ctx=ctx)
        finetune_net.collect_params().reset_ctx(ctx)
        finetune_net.hybridize()
        train_data = gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, num_workers=num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        test_data = gluon.data.DataLoader(
            gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
            'learning_rate': opt.lr, 'momentum': opt.momentum, 'wd': opt.wd})

        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        train_history = TrainingHistory(['training-error', 'validation-error'])

        iteration = 0
        lr_decay_count = 0
        best_decay_count = 0
        best_val_score = 0
        num_batch = len(train_data)


        for epoch in range(epochs):
            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate * lr_decay)
                lr_decay_count += 1
            tic = time.time()
            train_loss = 0
            train_metric.reset()


            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

                with ag.record():
                    output = [finetune_net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

                train_metric.update(label, output)
                name, acc = train_metric.get()
                iteration += 1

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            name, val_acc = test(finetune_net, ctx,val_data)
            train_history.update([1-acc, 1-val_acc])
            train_history.plot(save_path='%s/%s_history.png' % (plot_path, model_name))

            if val_acc > best_val_score:
                best_val_score = val_acc
                finetune_net.save_parameters('%s/%.4f-finetune-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

            logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
                (epoch, acc, val_acc, train_loss, time.time()-tic))

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                finetune_net.save_parameters('%s/finetune-%s-%d.params'%(save_dir, model_name, epoch))

            if epoch % 5 == 0:
                _, test_acc = test(finetune_net, ctx, test_data)
                logging.info('[Epoch %d] Test-acc: %.3f' % (epoch, test_acc))

        if save_period and save_dir:
            finetune_net.save_parameters('%s/finetune-%s-%d.params'%(save_dir, model_name, epochs-1))


    train(opt.num_epochs, context)


if __name__ == '__main__':
    main()















