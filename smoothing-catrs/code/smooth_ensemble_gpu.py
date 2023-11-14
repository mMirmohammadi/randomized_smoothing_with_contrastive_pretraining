'''
- this is the core file which supports ensembles for Randomized Smoothing
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/core.py written by Jeremy Cohen
'''

from typing import Iterable, List
import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class SmoothEnsemble(object):

    ABSTAIN = -1

    def __init__(self, base_classifiers, num_classes, sigma, aggregation_scheme, softmax_temp=None, softmax_idx=None):
        self.base_classifiers = base_classifiers
        self.num_classifiers = len(base_classifiers)
        self.num_classes = num_classes
        self.sigma = sigma
        self.aggregation_scheme = aggregation_scheme
        self.softmax_temp=softmax_temp
        self.softmax_idx=softmax_idx

    def certify(self, batch:Iterable[int], n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        
        counts_estimations_batch = self._sample_noises(batch, n, batch_size) # shape = (batch, 2*num_classifiers, num_classes)
        # print(counts_estimations_batch.shape)
        ret = []
        for i in range(len(batch)):
            counts_estimations = counts_estimations_batch[i]
            cAHats = [counts_estimation.argmax().item() for counts_estimation in counts_estimations]
            nAs = [counts_estimation[cAHat].item() for cAHat, counts_estimation in zip(cAHats, counts_estimations)]
            pABars = [self._lower_confidence_bound(nA, n, alpha) for nA in nAs]
            certified_radii = [(SmoothEnsemble.ABSTAIN, 0.0) if pABar < 0.5
                            else (cAHat, self.sigma * norm.ppf(pABar))
                            for pABar, cAHat in zip(pABars, cAHats)]
            ret.append(certified_radii)
        return ret

    def _sample_noises(self, batch: Iterable[int], num: int, batch_size) -> np.ndarray:
        outputs = [base_classifier[batch,:num].cuda() for base_classifier in self.base_classifiers]
        # outputs[0].shape = (len(batch), num, num_classes)
        predictions = self._get_predictions(outputs, num)
        
        return self._count_arr(predictions,self.num_classes)
        
        # with torch.no_grad():
        #     counts = np.zeros((self.num_classifiers*2, self.num_classes), dtype=int)
        #     for _ in range(ceil(num / batch_size)):
        #         this_batch_size = min(batch_size, num)
        #         num -= this_batch_size
                
        #         batch = x.repeat((this_batch_size, 1, 1, 1))
        #         batch = batch.to('cuda')
        #         noise = torch.randn_like(batch, device='cuda') * self.sigma
        #         inputs = batch+noise
        #         outputs = [base_classifier(inputs) for base_classifier in self.base_classifiers]
        #         for i, prediction in enumerate(predictions):
        #             counts[i] += self._count_arr(prediction.cpu().numpy(), self.num_classes)
        #     return counts

    # def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
    #     counts = np.zeros(length, dtype=int)
    #     for idx in arr:
    #         counts[idx] += 1
    #     return counts
    
    def _count_arr(self, arr: torch.Tensor, length: int) -> torch.Tensor:
        counts = []
        for cl in range(length):
            counts.append(torch.sum(arr == cl, dim=-1))
        counts = torch.stack(counts,dim=-1)
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
    # Returns the list of predictions of single models (0 mod 2) and ensemble models (1 mod 2)
    def _get_predictions(self, outputs: List[torch.Tensor], batch_size) -> List[torch.Tensor]:
        ensemble_outputs = torch.zeros_like(outputs[0])
        # ensemble_outputs = ensemble_outputs.to('cuda')
        cohen_weights = [0.0321, 0.1202, 0.0386, 0.0592, 0.1218, 0.0269, 0.0999, 0.0124, 0.1224, 0.1075]
        consistency_weights = [0.0229,  0.0568,  0.2844,  0.0997, -0.1398,  0.1760,  0.1653, -0.0079, 0.1170,  0.1249]
        predictions = []
        for i, output in enumerate(outputs): # for each base classifier
            if i == self.softmax_idx:
                output = torch.nn.functional.softmax(output.log()*self.softmax_temp, -1)
                
            if self.aggregation_scheme == 0: # soft-voting without softmax - default
                ensemble_outputs += output
            elif self.aggregation_scheme == 1: # hard voting
                # output [batch, N, num_cl]
                ensemble_outputs += torch.zeros_like(output).scatter_(-1,torch.argmax(output,-1,keepdim=True),1.)
                # if draw after hard voting, use randomness
                ensemble_outputs = ensemble_outputs + torch.rand_like(ensemble_outputs) * 0.1
            elif self.aggregation_scheme == 2: # soft-voting with/after softmax
                ensemble_outputs += torch.nn.functional.softmax(output, -1)
            elif self.aggregation_scheme == 3: # learned weights for pretrained Gaussian resnet110 (sigma=0.25)
                ensemble_outputs += output * cohen_weights[i]
            elif self.aggregation_scheme == 4: # learned weights for pretrained consistency resnet110 (sigma=0.25)
                ensemble_outputs += output * consistency_weights[i]
            elif self.aggregation_scheme == 5: # max pooling
                selection = output>ensemble_outputs
                ensemble_outputs[selection] = output[selection]
            
            predictions.append(output.argmax(-1)) # shape = (batch, N)
            predictions.append(ensemble_outputs.argmax(-1)) # shape = (batch, N)
            # print(predictions[-1].shape)

        return torch.stack(predictions,dim=-2)