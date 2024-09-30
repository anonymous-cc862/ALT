
import sklearn
import numbers

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing


import torch



import sys
sys.path.append('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/imbalanced_learn')

from imblearn.utils._docstring import _n_jobs_docstring
from imblearn.utils._docstring import Substitution
from imblearn.under_sampling.base import BaseCleaningSampler


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class TomekLinks1(BaseCleaningSampler):

    _parameter_constraints: dict = {
        **BaseCleaningSampler._parameter_constraints,
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(self, *, sampling_strategy="auto", n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_jobs = n_jobs

    @staticmethod
    def is_tomek(y, nn_index, class_type):
        """Detect if samples are Tomek's link.

        More precisely, it uses the target vector and the first neighbour of
        every sample point and looks for Tomek pairs. Returning a boolean
        vector with True for majority Tomek links.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not.

        nn_index : ndarray of shape (len(y),)
            The index of the closes nearest neighbour to a sample point.

        class_type : int or str
            The label of the minority class.

        Returns
        -------
        is_tomek : ndarray of shape (len(y), )
            Boolean vector on len( # samples ), with True for majority samples
            that are Tomek links.
        """
        links = np.zeros(len(y), dtype=bool)

        # find which class to not consider
        class_excluded = [c for c in np.unique(y) if c not in class_type]

        # there is a Tomek link between two samples if they are both nearest
        # neighbors of each others.
        for index_sample, target_sample in enumerate(y):
            if target_sample in class_excluded:
                continue

            if y[nn_index[index_sample]] != target_sample:
                if nn_index[nn_index[index_sample]] == index_sample:
                    links[index_sample] = True

        return links

    # def _fit_resample(self, X, y):
    #     # Find the nearest neighbour of every point
    #     nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
    #     nn.fit(X)
    #     nns = nn.kneighbors(X, return_distance=False)[:, 1]

    #     links = self.is_tomek(y, nns, self.sampling_strategy_)
    #     self.sample_indices_ = np.flatnonzero(np.logical_not(links))
    #     print(self.sample_indices_)

    #     return (
    #         _safe_indexing(X, self.sample_indices_),
    #         _safe_indexing(y, self.sample_indices_),
    #         self.sample_indices_
    #     )
    
    def _fit_resample(self, X, y):
        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=4, n_jobs=self.n_jobs) #5
        nn.fit(X)
        # nns = nn.kneighbors(X, return_distance=False)[:, 1]

        # links = self.is_tomek(y, nns, self.sampling_strategy_)
        # self.sample_indices_ = np.flatnonzero(np.logical_not(links))
        # print(self.sample_indices_)
        links=[0]* X.shape[0]
        for i in np.arange(1,3,1): #4
            nns1 = nn.kneighbors(X, return_distance=False)[:, i]


            links1 = self.is_tomek(y, nns1, self.sampling_strategy_) #true/false

            links+=links1


        self.sample_indices_ = np.flatnonzero(np.logical_not(links)) #false的index位置

        return (
            _safe_indexing(X, self.sample_indices_),
            _safe_indexing(y, self.sample_indices_),
            self.sample_indices_
        )

    def _more_tags(self):
        return {"sample_indices": True}




import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt


import torch

import lib.utils as utils
import lib.layers.odefunc as odefunc
from lib.custom_optimizers import Adam

import datasets

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular, override_divergence_fn
import random



import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt


import torch

import lib.utils as utils
import lib.layers.odefunc as odefunc
from lib.custom_optimizers import Adam

import datasets

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular, override_divergence_fn
import random



def setup_seed(seed):
    #seed(seed)
    torch.manual_seed(seed)
    #torch.manual_seed(0)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['mydata','power', 'gas', 'hepmass', 'miniboone', 'bsds300'], type=str, default='mydata'
)
parser.add_argument(
    "--layer_type", type=str, default='blend',
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)

parser.add_argument('--seed', type=int, default=777)
parser.add_argument('--epoch', type=int, default=1000)

parser.add_argument('--hdim_factor', type=int, default=1) 
parser.add_argument('--nhidden', type=int, default=5)
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.') 
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES) 

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS) #'dopri5'
parser.add_argument('--atol', type=float, default=1e-3)
parser.add_argument('--rtol', type=float, default=1e-3)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--early_stopping', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=None)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--val_freq', type=int, default=10)
parser.add_argument('--log_freq', type=int, default=10)

parser.add_argument('--scale', type=int, default=0.1)


args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0
    args.train_T = False

logger.info(args)

test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size

print('batch: ',args.batch_size)
def batch_iter(X, batch_size=args.batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


ndecs = 0


def update_lr(optimizer, n_vals_without_improvement):
    global ndecs
    if ndecs == 0 and n_vals_without_improvement > args.early_stopping // 3:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 10
        ndecs = 1
    elif ndecs == 1 and n_vals_without_improvement > args.early_stopping // 3 * 2:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 100
        ndecs = 2
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 10**ndecs


def load_data(name):
    if name == 'mydata':

        from datasets.mydata import MYDATA
        return MYDATA()

    if name == 'bsds300':
        return datasets.BSDS300()

    elif name == 'power':
        return datasets.POWER()

    elif name == 'gas':
        return datasets.GAS()

    elif name == 'hepmass':
        return datasets.HEPMASS()

    elif name == 'miniboone':
        return datasets.MINIBOONE()

    else:
        raise ValueError('Unknown dataset')


def compute_loss(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = model(x, zero)  # run model forward,model(x)得到z

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # log p(z) （suppose z服从高斯分布）
    logpx = logpz - delta_logp #loss是logpx 越大越好
    loss = -torch.mean(logpx) #max logpx
    return loss


def restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model

device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu") #3
if __name__ == '__main__':
    setup_seed(args.seed)
    device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu") #3 5
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    logger.info('Using {} GPUs.'.format(torch.cuda.device_count())) 

    data = load_data(args.data) #no noise
    data.trn.x = torch.from_numpy(data.trn.x)  #x 训练的是
    data.val.x = torch.from_numpy(data.val.x)
    #data.tst.x = torch.from_numpy(data.tst.x)

    args.dims = '-'.join([str(args.hdim_factor * data.n_dims)] * args.nhidden)
    #'-'.join([str(10 * 256)] * 1)  即 2560

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    setup_seed(args.seed)
    model = build_model_tabular(args, data.n_dims, regularization_fns).to(device) #data.n_dims是网络output dim
    set_cnf_options(args, model)

    for k in model.state_dict().keys():
        logger.info(k)

    if args.resume is not None:
        checkpt = torch.load(args.resume)

        # Backwards compatibility with an older version of the code.
        # TODO: remove upon release.
        filtered_state_dict = {}
        for k, v in checkpt['state_dict'].items():
            if 'diffeq.diffeq' not in k:
                filtered_state_dict[k.replace('module.', '')] = v 
        model.load_state_dict(filtered_state_dict)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))
    setup_seed(args.seed)
    if not args.evaluate:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        time_meter = utils.RunningAverageMeter(0.98)
        loss_meter = utils.RunningAverageMeter(0.98)
        nfef_meter = utils.RunningAverageMeter(0.98)
        nfeb_meter = utils.RunningAverageMeter(0.98)
        tt_meter = utils.RunningAverageMeter(0.98)

        best_loss = float('inf')
        itr = 0
        n_vals_without_improvement = 0
        end = time.time()
        model.train()

        epo=[]
        loss_history=[]
        for ep in range(args.epoch):
            loss_batch=[]
            epo.append(ep+1)
            print('ep', ep)
            if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping: 
                break
            #it=0
            for x in batch_iter(data.trn.x, shuffle=True):

                if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping:
                    break
            
                optimizer.zero_grad()

                x = cvt(x) 
                loss = compute_loss(x, model) 
                loss_meter.update(loss.item()) 

                if len(regularization_coeffs) > 0:
                    reg_states = get_regularization(model, regularization_coeffs)
                    reg_loss = sum(
                        reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                    )
                    loss = loss + reg_loss

                total_time = count_total_time(model)
                nfe_forward = count_nfe(model)

                loss_batch.append(loss.item())

                loss.backward()
                optimizer.step()

                nfe_total = count_nfe(model)
                nfe_backward = nfe_total - nfe_forward
                nfef_meter.update(nfe_forward)
                nfeb_meter.update(nfe_backward)

                time_meter.update(time.time() - end)
                tt_meter.update(total_time)

                if itr % args.log_freq == 0:
                    log_message = (
                        'Iter {:06d} | Epoch {:.2f} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | '
                        'NFE Forward {:.0f}({:.1f}) | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                            itr,
                            float(itr) / (data.trn.x.shape[0] / float(args.batch_size)), time_meter.val, time_meter.avg,
                            loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg, nfeb_meter.val,
                            nfeb_meter.avg, tt_meter.val, tt_meter.avg
                        )
                    )
                    if len(regularization_coeffs) > 0:
                        log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)

                    logger.info(log_message)
                itr += 1
                end = time.time()

                # Validation loop.
                if itr % args.val_freq == 0:
                    model.eval()
                    start_time = time.time()
                    with torch.no_grad():
                        val_loss = utils.AverageMeter()
                        val_nfe = utils.AverageMeter()
                        for x in batch_iter(data.val.x, batch_size=test_batch_size):
                            x = cvt(x)
                            val_loss.update(compute_loss(x, model).item(), x.shape[0])
                            val_nfe.update(count_nfe(model))

                        if val_loss.avg < best_loss:
                            best_loss = val_loss.avg
                            utils.makedirs(args.save)
                            torch.save({
                                'args': args,
                                'state_dict': model.state_dict(),
                            }, os.path.join(args.save, 'checkpt.pth'))
                            n_vals_without_improvement = 0
                        else:
                            n_vals_without_improvement += 1
                        update_lr(optimizer, n_vals_without_improvement)

                        log_message = (
                            '[VAL] Iter {:06d} | Val Loss {:.6f} | NFE {:.0f} | '
                            'NoImproveEpochs {:02d}/{:02d}'.format(
                                itr, val_loss.avg, val_nfe.avg, n_vals_without_improvement, args.early_stopping
                            )
                        )
                        logger.info(log_message)
                    model.train()

            loss_history.append(np.mean(loss_batch))


        logger.info('Training has finished.')
        model = restore_model(model, os.path.join(args.save, 'checkpt.pth')).to(device)
        
        set_cnf_options(args, model)

    model.eval()


    #generate new samples
    setup_seed(args.seed)
    data_shape= (data.n_dims,) 
    new=torch.tensor(np.random.normal(loc=0, scale=args.scale, size=(data.new_num, *data_shape))) 
    fixed_z = cvt(new)
    with torch.no_grad():      
        generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape) 



###tomek
    x_all=np.loadtxt('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/train_all_emb_new.txt',dtype=np.float32)
    y_all=np.loadtxt('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/train_all_targets.txt',dtype=np.float32)
    
    print('x_normal',x_all.shape)
    
    z_all=model(cvt(torch.from_numpy(x_all)), reverse=False) 

    z_abnormal_new=model(generated_samples, reverse=False) 

    z_train_embeddings=torch.cat([z_all,z_abnormal_new], dim=0)
    z_train_labels=torch.tensor(np.append(y_all,[1]*z_abnormal_new.shape[0],axis=0))

    tl = TomekLinks1(sampling_strategy='all')#,random_state=42)
    z_train_embeddings_new, z_train_labels_new,ind = tl.fit_resample(z_train_embeddings.detach().to('cpu'), z_train_labels)
    print('indddd',len(ind))

    np.savetxt('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/ffjord/generated_samples_new1.txt', generated_samples.cpu())
    np.savetxt('/home/rliuaj/fraud_contrastive/ITGAN-main_copy2_copy6/ffjord/ind.txt', ind)
    print(ind.shape)
    print('tabular1 finish')

