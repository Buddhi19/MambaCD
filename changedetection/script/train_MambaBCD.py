import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname((os.path.dirname(__file__))))))

import argparse
import os
import time

import numpy as np

from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_optimizer as topt
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.MambaBCD import STMambaBCD

import MambaCD.changedetection.utils_func.lovasz_loss as L

from MambaCD.changedetection.utils_func.loss import ce2_dice1
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.evaluator = Evaluator(num_class=2)

        self.deep_model = STMambaBCD(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
        self.deep_model = self.deep_model.cuda()
        file_name = input("Input the file name for saving the model: ")
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + file_name)
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = self._make_optimizer()
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_save_path, 'logs'))

    def _make_optimizer(self):
        name = self.args.optimizer.lower()

        if name == "adamw":
            return optim.AdamW(
                self.deep_model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        if name == "sgd":
            return optim.SGD(
                self.deep_model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
                nesterov=True,
            )

        # ---- mapping of torch-optimizer aliases to their classes ----
        opt_table = {
            # adaptive Adam variants
            "adabelief":   topt.AdaBelief,
            "adabound":    topt.AdaBound,
            "adamod":      topt.AdaMod,
            "adamp":       topt.AdamP,
            "adahessian":  topt.Adahessian,
            "radam":       topt.RAdam,
            "qhadam":      topt.QHAdam,
            "adam":        topt.AdamP,           # convenience alias
            # transformers / large-batch
            "lamb":        topt.Lamb,
            "lars":        topt.LARS if hasattr(topt, "LARS") else None,
            # second-order & curvature-aware
            "shampoo":     topt.Shampoo,
            # lookahead families
            "lookahead":   topt.Lookahead,
            "ranger":      topt.Ranger,
            "rangerqh":    topt.RangerQH,
            "rangerva":    topt.RangerVA,
            # accelerated SGD variants
            "sgdw":        topt.SGDW,
            "sgdp":        topt.SGDP,
            "accsgd":      topt.AccSGD,
            # gradient rescaling / momentum tricks
            "madgrad":     topt.MADGRAD,
            "diffgrad":    topt.DiffGrad,
            "aggmo":       topt.AggMo,
            "apollo":      topt.Apollo,
            "pid":         topt.PID,
            # A2Grad family
            "a2gradexp":   topt.A2GradExp,
            "a2gradinc":   topt.A2GradInc,
            "a2graduni":   topt.A2GradUni,
            # others
            "novograd":    topt.NovoGrad,
            "yogi":        topt.Yogi,
            "swats":       topt.SWATS,
            "sgdw":        topt.SGDW,
            "sgdp":        topt.SGDP,
            "qhm":         topt.QHM,
            "adadelta":    topt.Adafactor,       # alias (name clash with torch)
            "adadfactor":  topt.Adafactor,
        }

        opt_cls = opt_table.get(name, None)
        if opt_cls is None:
            raise ValueError(f"Unsupported optimizer '{self.args.optimizer}'. "
                            f"Check spelling or update opt_table.")

        # Most torch-optimizer classes accept (params, lr, weight_decay, **kwargs).
        # Pass only the common arguments here; tune others in the CLI/config if needed.
        return opt_cls(
            self.deep_model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()
            ce_loss_1 = ce2_dice1(output_1, labels)
            lovasz_loss = L.lovasz_softmax(F.softmax(output_1, dim=1), labels, ignore=255)
            

            final_loss = ce_loss_1 + 0.5*lovasz_loss

            final_loss.backward()
            self.optim.step()

            self.writer.add_scalar('CDLoss/train', final_loss.item(), itera + 1)

            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, overall loss is {final_loss}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    self.writer.add_scalar('CDMetrics/Recall', rec, itera + 1)
                    self.writer.add_scalar('CDMetrics/Precision', pre, itera + 1)
                    self.writer.add_scalar('CDMetrics/OA', oa, itera + 1)
                    self.writer.add_scalar('CDMetrics/F1_score', f1_score, itera + 1)
                    self.writer.add_scalar('CDMetrics/IoU', iou, itera + 1)
                    self.writer.add_scalar('CDMetrics/Kappa', kc, itera + 1)
                    if kc > best_kc and oa> 0.92:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model_{oa}_{kc}.pth'))
                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)
        self.writer.close()

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset = ChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=4, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels, _ = data
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda()
                labels = labels.cuda().long()

                output_1 = self.deep_model(pre_change_imgs, post_change_imgs)

                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()

                self.evaluator.add_batch(labels, output_1)
                
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Recall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc

def main():
    parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD+/WHU-CD dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='SYSU')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/train')
    parser.add_argument('--train_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
