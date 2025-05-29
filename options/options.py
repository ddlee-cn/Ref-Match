import argparse
import os
from util import util
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='dev/example/src', help='name of the experiment')
        self.parser.add_argument('--ref_name', type=str, default='dev/example/trg', help='name of the reference image')
        self.parser.add_argument('--mask_type', type=str, default='box', help='mask type: box or irr')
        self.parser.add_argument('--data_root', type=str, default='data', help='data root directory')
        self.parser.add_argument('--in_root', type=str, default='data', help='in data root directory')
        self.parser.add_argument('--imageSize', type=int, default=256, help='rescale the image to this size')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--tau', type=float, default=0.05, help='response threshold')
        self.parser.add_argument('--k_per_level', type=float, default=float('inf'), help='maximal number of best buddies per local search.')
        self.parser.add_argument('--results_dir', type=str, default='./results', help='models are saved here')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')        
        # for VGG deconv
        self.parser.add_argument('--input_nc', type=int, default=3, help='number of input channels')
        self.parser.add_argument('--batchSize', type=int, default=1, help='batch size')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate for adam')
        self.parser.add_argument('--gamma', type=float, default=1, help='weight for equallibrium in BEGAN or ratio between I0 and Iref features for optimize_based_features')
        self.parser.add_argument('--convergence_threshold', type=float, default=0.001, help='threshold for convergence for watefall mode (for optimize_based_features_model)')
        self.parser.add_argument('--pretrained_path', default='./results/vgg19.pth', help='default pretrained model file path')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
