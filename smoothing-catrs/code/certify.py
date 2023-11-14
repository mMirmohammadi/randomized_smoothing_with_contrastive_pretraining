# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

""" Evaluate a smoothed classifier on a dataset. """
import argparse
import os
import datetime
from time import time
import numpy as np

#import setGPU
import torch

from third_party.core import Smooth
from architectures import get_architecture
from datasets import get_dataset, DATASETS, get_num_classes
from train_utils import AverageMeter, accuracy, log, test
from train_utils import prologue

from tqdm import trange

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--start", type=int, default=0, help='start')
parser.add_argument('--save-predictions', action='store_true', help='Whether to store the model predictions for each noisy sample.')
parser.add_argument("--load-noise", type=str, default=None, help="File with noise to add to each sample (Optional).")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    print(checkpoint.keys())
    # exit(0)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['model'])
    base_classifier.cuda()
    print(base_classifier)
    method_name = args.base_classifier.split('/')[2]
    print('method name:', method_name)

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    acc_meter = AverageMeter()
    rad_meter = AverageMeter()

    if args.load_noise is not None:
        smoothed_classifier.load_noise(args.load_noise)
        use_loaded_noise=True

    # prepare output file
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # if os.path.exists(args.outfile):
    #     raise 'file already exists'
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    all_predictions = []
    pbar = trange(args.start, len(dataset))
    rads = []
    for i in pbar:

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        if args.save_predictions and i>0 and i%1000==0:
            pred_path = f'/mnt/disks/rs-cert-data/predictions/{args.dataset}/{smoothed_classifier.mode}/{args.sigma}/predictions_{args.N}_{method_name}_2l.pt'
            if not os.path.exists(os.path.dirname(pred_path)):
                os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            print(pred_path)
            all_predictions_t = torch.stack(all_predictions)
            print(all_predictions_t.shape)
            torch.save(all_predictions_t,pred_path)

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        ret = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch, 
                                                         return_predictions=args.save_predictions,
                                                         use_loaded_noise=use_loaded_noise)
        prediction = ret[0]
        radius = ret[1]
        if args.save_predictions:
            predictions = ret[2]
            all_predictions.append(predictions.cpu())
        after_time = time()
        correct = int(prediction == label)
        if not correct: radius = 0.
        rads.append(radius)
        acc_meter.update(correct)
        rad_meter.update(radius)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3f}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        postfix_str = f"ACC: {acc_meter.avg:.3f}, ACR: {rad_meter.avg:.3f}"

        pbar.set_postfix_str(postfix_str)

    rads = np.array(rads)
    for thr in [0.25*i for i in range(11)]:
        acc = sum(rads>thr)
        if acc > 0:
            print(f'{thr:.2f}\t{acc/100:.2f}')

    f.close()
    if args.save_predictions:
        pred_path = f'/mnt/disks/rs-cert-data/predictions/{args.dataset}/{smoothed_classifier.mode}/{args.sigma}/predictions_{args.N}_{method_name}_2l.pt'
        if not os.path.exists(os.path.dirname(pred_path)):
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        print(pred_path)
        all_predictions = torch.stack(all_predictions)
        print(all_predictions.shape)
        torch.save(all_predictions,pred_path)
