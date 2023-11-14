'''
- this is the file which does certification for the SmoothEnsemble class (smooth_ensemble.py)
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/certify.py written by Jeremy Cohen
'''

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from smooth_ensemble_gpu import SmoothEnsemble
from time import time
import torch
import datetime
# from architectures import get_architecture
# from architectures import get_architecture_center_layer

import random
import numpy as np
from tqdm import trange

from combinations import combinations

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("base_classifiers", type=str, help="path to saved predictions of base classifiers", nargs='+')
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--skip_offset", type=int, default=0, help="which mod to consider while skipping")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--start", type=int, default=0, help='start')
parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
parser.add_argument("--aggregation-scheme", type=int, default=0, help="0 is default softvoting; 1 is hard voting; 2 is softvoting after softmax, 3 and 4 are weightings according to prelearned weights for specific models")
parser.add_argument("--combinations", type=int, default=-1, help="Number of models to combine in ensamble. All available combinations of this size will be evaluated."
                    "Defaults to -1, equivalent to taking all models in ensamble (only one evaluation is necessary).")
parser.add_argument("--combinations-to-analyze", type=str, default=None, nargs='*', help="Which Combinations of models to annalyze. Overrides args.combinations.")
parser.add_argument("--comb-dict", type=str, default="", help="names to give to each model")
parser.add_argument("--center-layer", type=int, default=0, help="set to 1 if model requires centering the layer")
parser.add_argument('--softmax-temp', default=None,  type=float, nargs='*', help='Temperatures to use for softmax on SVM trained models')
parser.add_argument('--softmax-idx', default=None,  type=int, help='Index of the model for which to apply softmax temp.')
args = parser.parse_args()

seed = args.seed
print('seed: ', seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.set_printoptions(precision=10)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(args.softmax_temp)
print(args.softmax_idx)
# exit(0)

if __name__ == "__main__":

    # load the base classifiers
    all_base_classifiers = []
    # for PATH in args.base_classifiers:
    #     checkpoint = torch.load(PATH)
    #     base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    #     if args.center_layer > 0:
    #         base_classifier = get_architecture_center_layer(checkpoint["arch"], args.dataset)
    #     base_classifier.load_state_dict(checkpoint['state_dict'])
    #     base_classifiers.append(base_classifier)
    num_samples = 1e8
    for path in args.base_classifiers:
        print('loading file',path)
        start = time()
        preds:torch.Tensor = torch.load(path)
        # preds:torch.Tensor = torch.zeros((3,3,3))
        print('loaded file', path)
        print('Time elapsed', time()-start)
        print(preds.shape)
        all_base_classifiers.append(preds)
        num_samples = min(num_samples, preds.shape[0])

    dataset = get_dataset(args.dataset, args.split)
    
    if args.combinations != -2:
        combinations_to_analyze = combinations(len(all_base_classifiers), args.combinations)
    else:
        combinations_to_analyze = []
        for k in range(2,len(all_base_classifiers)+1):
            combinations_to_analyze += combinations(len(all_base_classifiers), k)
    if args.combinations_to_analyze is not None:
        combinations_to_analyze = [[int(x) for x in cmb] for cmb in args.combinations_to_analyze]

    if args.comb_dict:
        comb_dict = {i:args.comb_dict[i] for i in range(len(all_base_classifiers))}
    else:
        comb_dict = {i:str(i) for i in range(len(all_base_classifiers))}

    cmb_strs = [''.join([comb_dict[x] for x in cmb]) for cmb in combinations_to_analyze]

    for ci, cmb in enumerate(combinations_to_analyze): 
        # break
        cmb_str = ''.join([comb_dict[x] for x in cmb])
        print(ci+1, '/', len(combinations_to_analyze), cmb, cmb_str)
        # continue
        # if (7 not in cmb) and (8 not in cmb): continue
        base_classifiers = [all_base_classifiers[i] for i in cmb]
        softmax_temps=[1.]
        softmax_idx = None
        if args.softmax_idx is not None and args.softmax_temp is not None and args.softmax_idx in cmb:
            softmax_temps = args.softmax_temp
            softmax_idx = args.softmax_idx

        for T in softmax_temps:    
            # create the smooothed ensemble classifier g
            smoothed_classifier = SmoothEnsemble(base_classifiers, get_num_classes(args.dataset), args.sigma, args.aggregation_scheme, softmax_temp=T, softmax_idx=softmax_idx)

            # prepare output file
            output_files = []
            output_dir = os.path.dirname(args.outfile)
            output_fn = os.path.basename(args.outfile)
            ext = output_fn.split('.')[-1]
            if ext == output_fn: ext = ''
            else: 
                ext = f'.{ext}'
                output_fn = output_fn.replace(ext,'')
            if softmax_idx is not None:
                output_fn = output_fn + f'_T{T:.1f}'

            output_dir = os.path.join(output_dir,cmb_str)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, output_fn)
            print('',output_file)
            # continue
            for i in range(len(base_classifiers)*2):
                f = open(output_file+"_"+str(i)+ext, 'w')
                print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
                output_files.append(f)
            with open(output_file + '_models.txt','w') as f:
                for i in cmb:
                    print(f"{i} {args.base_classifiers[i]}", file=f, flush=True)

            results_list = []
            for i in range(len(base_classifiers)*2):
                results_list.append([])

            # iterate through the dataset
            pbar = trange(args.start, (num_samples-1)//args.batch+1)
            rads = []
            for i in pbar:

                this_batch = [*range(i*args.batch, min(num_samples,(i+1)*args.batch))]
                # only certify every args.skip examples, and stop after args.max examples
                if i % args.skip != args.skip_offset:
                    continue
                if args.max in this_batch:
                    break

                before_time = time()
                xs = []
                labels = []
                for jj in this_batch:
                    (x, label) = dataset[jj]
                    xs.append(x)
                    labels.append(label)
                # certify the predictions of g around x
                # x = x.cuda()
                certification_results_batch = smoothed_classifier.certify(this_batch, args.N0, args.N, args.alpha, args.batch)
                after_time = time()
                time_elapsed = datetime.timedelta(seconds=(after_time - before_time))

                for idx in range(len(this_batch)):
                    certification_results = certification_results_batch[idx]
                    label = labels[idx]

                    for j, f in enumerate(output_files):
                        prediction = certification_results[j][0]
                        correct = int(prediction == label)
                        radius = certification_results[j][1]
                        
                        # approximates running times, assuming that all individual models approximately require the same running time
                        current_time_elapsed = time_elapsed
                        if j % 2 == 0:
                            current_time_elapsed /= len(base_classifiers)
                        else:
                            current_time_elapsed = ((j+1)/2) / len(base_classifiers) * time_elapsed
                        current_time_elapsed = str(current_time_elapsed)
                        
                        results_list[j].append([i, label, prediction, radius, correct, current_time_elapsed])
                        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                            i, label, prediction, radius, correct, current_time_elapsed), file=f, flush=True)
                        
                        # if len(results_list[j]) % 100 == 0:
                        #     np.save(output_file+"_"+str(j), np.array(results_list[j]))
                    
                # for j in range(len(base_classifiers)*2):
                #     np.save(output_file+"_"+str(j), np.array(results_list[j]))

            for f in output_files: f.close()
        
    print(cmb_strs)