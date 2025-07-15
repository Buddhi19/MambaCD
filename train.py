import os
import sys
import torch
from dotenv import load_dotenv

load_dotenv()
def getPath(env_path):
    return os.path.expanduser(os.getenv(env_path))

VSSM_MODEL_PATH = getPath('VSSMBASEPATH')
model_path = getPath('MODELPATH')

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

torch.cuda.set_device(1)

from MambaCD.changedetection.script import train_MambaBCD

configs_path = os.path.join(main_dir, 'MambaCD/changedetection/configs/vssm1/vssm_base_224.yaml')


LEVIR_dataset_path = getPath('LEVIRCDPATH')

LEVIR_train_dataset_path = os.path.join(LEVIR_dataset_path, 'train')
LEVIR_test_dataset_path = os.path.join(LEVIR_dataset_path, 'test')
LEVIR_train_data_list_path = os.path.join(LEVIR_dataset_path, 'train.txt')
LEVIR_test_data_list_path = os.path.join(LEVIR_dataset_path, 'test.txt')

SYSU_dataset_path = getPath('SYSUCDPATH')

SYSU_train_dataset_path = os.path.join(SYSU_dataset_path, 'train')
SYSU_test_dataset_path = os.path.join(SYSU_dataset_path, 'test')
SYSU_train_data_list_path = os.path.join(SYSU_dataset_path, 'train.txt')
SYSU_test_data_list_path = os.path.join(SYSU_dataset_path, 'test.txt')

WHU_dataset_path = getPath('WHUCDPATH')

WHU_train_dataset_path = os.path.join(WHU_dataset_path, 'train')
WHU_test_dataset_path = os.path.join(WHU_dataset_path, 'test')
WHU_train_data_list_path = os.path.join(WHU_dataset_path, 'train.txt')
WHU_test_data_list_path = os.path.join(WHU_dataset_path, 'test.txt')

train_data_list = []
with open(LEVIR_train_data_list_path, 'r') as f:
    for line in f:
        train_data_list.append(line.strip())

test_data_list = []
with open(LEVIR_test_data_list_path, 'r') as f:
    for line in f:
        test_data_list.append(line.strip())

class ARGS:
    def __init__(self):
        self.dataset_path = LEVIR_dataset_path
        self.pretrained_weight_path = VSSM_MODEL_PATH
        self.dataset = 'LEVIR-CD+'
        self.opts = None
        self.type = 'train'
        self.shuffle = True
        self.crop_size = 256
        self.batch_size = 4
        self.max_iters = 50000
        self.start_iter = 0
        self.cuda = True
        self.model_type = 'MambaBCD'
        self.optimizer = 'adamw'
        self.learning_rate = 1e-4
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.train_dataset_path = LEVIR_train_dataset_path
        self.train_data_name_list = train_data_list
        self.test_dataset_path = LEVIR_test_dataset_path
        self.test_data_name_list = test_data_list
        self.cfg = configs_path
        self.model_param_path = model_path
        self.resume = None

        self.model_saving_name = 'model name'

def LEVIR_main():
    args = ARGS()
    trainer_LEVIR = train_MambaBCD.Trainer(args)
    trainer_LEVIR.training()
    trainer_LEVIR.validation()

def SYSU_main():
    args = ARGS()
    args.dataset = 'SYSU'
    args.model_saving_name = 'MambaBCD_SYSU_base'
    args.dataset_path = SYSU_dataset_path
    args.train_dataset_path = SYSU_train_dataset_path
    args.train_data_name_list = []
    with open(SYSU_train_data_list_path, 'r') as f:
        for line in f:
            args.train_data_name_list.append(line.strip())
    args.test_dataset_path = SYSU_test_dataset_path
    args.test_data_name_list = []
    with open(SYSU_test_data_list_path, 'r') as f:
        for line in f:
            args.test_data_name_list.append(line.strip())
    trainer_SYSU = train_MambaBCD.Trainer(args)
    trainer_SYSU.training()
    trainer_SYSU.validation()

def WHU_main():
    args = ARGS()
    args.dataset = 'WHU-CD'
    args.model_saving_name = 'MambaBCD_WHU_base'
    args.dataset_path = WHU_dataset_path
    args.train_dataset_path = WHU_train_dataset_path
    args.train_data_name_list = []
    with open(WHU_train_data_list_path, 'r') as f:
        for line in f:
            args.train_data_name_list.append(line.strip())
    args.test_dataset_path = WHU_test_dataset_path
    args.test_data_name_list = []
    with open(WHU_test_data_list_path, 'r') as f:
        for line in f:
            args.test_data_name_list.append(line.strip())
    trainer_WHU = train_MambaBCD.Trainer(args)
    trainer_WHU.training()
    trainer_WHU.validation()

if __name__ == "__main__":
    SYSU_main()