# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

from typing import *
import math

import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()

    def acr(self):
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return (df["correct"] * df["radius"]).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def smallplot_certified_accuracy(outfile: str, title: str, max_radius: float,
                                 methods: List[Line], radius_step: float = 0.01, xticks=0.5) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for method in methods:
        plt.plot(radii, method.quantity.at_radii(radii), method.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.tick_params(labelsize=20)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))
    plt.legend([method.legend for method in methods], loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()


def latex_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')

    for radius in radii:
        f.write("& $r = {:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for i, method in enumerate(methods):
        f.write(method.legend)
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = r" & \textbf{" + "{:.2f}".format(accuracies[i, j]) + "}"
            else:
                txt = " & {:.2f}".format(accuracies[i, j])
            f.write(txt)
        f.write("\\\\\n")
    f.close()


def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii) + 1))
    for i, method in enumerate(methods):
        accuracies[i, :-1] = method.quantity.at_radii(radii)
        accuracies[i, -1] = method.quantity.acr()
    
    f = open(outfile, 'w')
    f.write("|  | acr |")
    for radius in radii:
        f.write("r = {:.3} |".format(radius))
    f.write("\n")

    f.write("| --- | --- |")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        print(method.legend)
        f.write("<b> {} </b>| ".format(method.legend))
        if i == accuracies[:, -1].argmax():
            txt = "{:.3f}<b>*</b> |".format(accuracies[i, -1])
        else:
            txt = "{:.3f} |".format(accuracies[i, -1])
        f.write(txt)
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.3f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.3f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()


if __name__ == '__main__':
    from combinations import _combinations
    cmbs = list(reversed([cm[::-1] for cm in _combinations(9,3)]))
    cmbs = ['03', '04', '34', '05', '35', '45', '07', '37', '47', '57', '034', '035', '045', '345', '037', '047', '347', '057', '357', '457', '0345', '0347', '0357', '0457', '3457', '03457']
    cmbs = ['047', '057', '0457']
    temps = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.5, 3, 4, 5]
    tcmbs = []
    for cmb in cmbs:
        for tmp in temps:
            tcmbs.append((cmb,tmp))
    # print(cmbs)

    methods = []
    file_names_25 = [
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/cohen/256/0/noise_0.25.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/stab/lbd_2.0/256/0/noise_0.25.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/salman/pgd_256.0_10_10/num_4/256/0/noise_0.25.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/macer/num_16/lbd_16.0/gamma_8.0/beta_16.0/64/2/noise_0.25.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/consistency/cohen/num_2/lbd_20.0/eta_0.5/noise_0.25.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/smix_0.5_4_m0/eta_5.0/num_2/256/0/noise_0.25.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/catrs/adv_256.0_4/lbd_0.5/num_4/0/noise_0.25.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/017/noise_0.25_4.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/018/noise_0.25_4.tsv',
        # '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/018/noise_0.25_0.tsv',
        # '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/012/noise_0.25_0.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/catrs_macer/2/noise_0.25.tsv_3',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/noise_0.25.tsv_13',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/5/noise_0.25.tsv_13',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/0478/noise_0.25_7.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/0345/noise_0.25_7.tsv',
    ] + [f'/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/0457/0/{cmb}/noise_0.25_T{tmp:.1f}_{2*len(cmb)-1}.tsv' for cmb, tmp in tcmbs]
    names_25 = [
        'cohen', 
        'stability', 
        'salman', 
        'macer', 
        'consistency', 
        'smoothmix', 
        'catrs',
        'supcon_2l',
        'supcon_3l',
        # 'cohen_again',
        # 'cohen_again2',
        'catrs_macer',
        'all_0',
        'all_5',
        'all_0478',
        'all_0345',
    ] + [f'all_{cmb}_T{tmp:.1f}' for cmb, tmp in tcmbs]

    cmbs = ['012', '013', '023', '123', '014', '024', '124', '034', '134', '234', '015', '025', '125', '035', '135', '235', '045', '145', '245', '345', '016', '026', '126', '036', '136', '236', '046', '146', '246', '346', '056', '156', '256', '356', '456', '017', '027', '127', '037', '137', '237', '047', '147', '247', '347', '057', '157', '257', '357', '457', '067', '167', '267', '367', '467', '567'] + [
        '0123', '0124', '0134', '0234', '1234', '0125', '0135', '0235', '1235', '0145', '0245', '1245', '0345', '1345', '2345', '0126', '0136', '0236', '1236', '0146', '0246', '1246', '0346', '1346', '2346', '0156', '0256', '1256', '0356', '1356', '2356', '0456', '1456', '2456', '3456', '0127', '0137', '0237', '1237', '0147', '0247', '1247', '0347', '1347', '2347', '0157', '0257', '1257', '0357', '1357', '2357', '0457', '1457', '2457', '3457', '0167', '0267', '1267', '0367', '1367', '2367', '0467', '1467', '2467', '3467', '0567', '1567', '2567', '3567', '4567']
    file_names_50 = [
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/cohen/256/0/noise_0.5.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/stab/lbd_2.0/256/0/noise_0.5.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/salman/pgd_256.0_10_10/num_8/256/0/noise_0.5.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/macer/num_16/lbd_4.0/gamma_8.0/beta_16.0/64/2/noise_0.5.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/consistency/cohen/num_2/lbd_10.0/eta_0.5/noise_0.5.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/smix_1.0_4_m1/eta_5.0/num_2/256/0/noise_0.5.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/catrs/adv_256.0_4/lbd_1.0/num_4/0/noise_0.5.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/0.5/0457/noise_0.5_6.tsv'
    ] + [f'/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/0.5/{cmb}/noise_0.5_{2*len(cmb)-1}.tsv' for cmb in cmbs]
    names_50 = [
        'cohen', 
        'stability', 
        'salman', 
        'macer', 
        'consistency', 
        'smoothmix', 
        'catrs',
        'supcon_2l',
        # 'supcon_3l',
        # 'cohen_again',
        # 'cohen_again2',
        # 'catrs_macer',
        # 'all_0',
        # 'all_5',
        # 'all_0478',
        # 'all_0345',
    ] + [f'all_{cmb}' for cmb in cmbs]

    cmbs = ['012', '014', '024', '124', '015', '025', '125', '045', '145', '245', '016', '026', '126', '046', '146', '246', '056', '156', '256', '456', '017', '027', '127', '047', '147', '247', '057', '157', '257', '457', '067', '167', '267', '467', '567'] + [
        '0124', '0125', '0145', '0245', '1245', '0126', '0146', '0246', '1246', '0156', '0256', '1256', '0456', '1456', '2456', '0127', '0147', '0247', '1247', '0157', '0257', '1257', '0457', '1457', '2457', '0167', '0267', '1267', '0467', '1467', '2467', '0567', '1567', '2567', '4567']
    file_names_100 = [
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/cohen/256/0/noise_1.0.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/stab/lbd_1.0/256/0/noise_1.0.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/salman/pgd_512.0_10_10/num_4/256/1/noise_1.0.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/macer_deferred50/num_16/lbd_12.0/gamma_8.0/beta_16.0/64/2/noise_1.0.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/consistency/cohen/num_2/lbd_10.0/eta_0.5/noise_1.0.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/smix_2.0_4_m1/eta_5.0/num_2/256/0/noise_1.0.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/catrs/adv_256.0_4/lbd_2.0/num_4/0/noise_1.0.tsv',
        '/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/1.0/017/noise_1.0_4.tsv'
    ] + [f'/home/stefan_balauca/rs-cpt/smoothing-catrs/test/certify/cifar10/ensamble/all/0/1.0/{cmb}/noise_1.0_{2*len(cmb)-1}.tsv' for cmb in cmbs]
    names_100 = [
        'cohen', 
        'stability', 
        'salman', 
        'macer', 
        'consistency', 
        'smoothmix', 
        'catrs',
        'supcon_2l',
        # 'supcon_3l',
        # 'cohen_again',
        # 'cohen_again2',
        # 'catrs_macer',
        # 'all_0',
        # 'all_5',
        # 'all_0478',
        # 'all_0345',
    ] + [f'all_{cmb}' for cmb in cmbs]

    print(len(file_names_25), len(names_25), len([f'all_{cmb}' for cmb in cmbs]))
    for fn, nm in zip(file_names_25, names_25):
        acc = ApproximateAccuracy(fn)
        ln = Line(acc,legend=f'{nm} 0.25')
        methods.append(ln)
        print(nm)

    markdown_table_certified_accuracy('test/results/cert_0.25.md', 0, 1, 0.25, methods)
    
    methods = []
    for fn, nm in zip(file_names_50, names_50):
        acc = ApproximateAccuracy(fn)
        ln = Line(acc,legend=f'{nm} 0.50')
        methods.append(ln)
    markdown_table_certified_accuracy('test/results/cert_0.5.md', 0, 2, 0.25, methods)
    
    methods = []
    for fn, nm in zip(file_names_100, names_100):
        acc = ApproximateAccuracy(fn)
        ln = Line(acc,legend=f'{nm} 1.0')
        methods.append(ln)
    markdown_table_certified_accuracy('test/results/cert_1.0.md', 0, 4, 0.25, methods)

    # micromamba activate catrs