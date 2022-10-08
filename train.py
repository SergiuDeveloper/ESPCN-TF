import importlib
import utils
import model
import neuralnet
importlib.reload(utils)
importlib.reload(model)
importlib.reload(neuralnet)

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from utils.dataset import dataset
from utils.common import PSNR
from model import ESPCN 
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=500000, help='-')
parser.add_argument("--scale",          type=int, default=4,      help='-')
parser.add_argument("--batch-size",     type=int, default=128,    help='-')
parser.add_argument("--save-every",     type=int, default=1000,    help='-')
parser.add_argument("--save-best-only", type=int, default=0,      help='-')
parser.add_argument("--save-log",       type=int, default=0,      help='-')
parser.add_argument("--ckpt-dir",       type=str, default="",     help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAG, unparsed = parser.parse_known_args()
steps = FLAG.steps
batch_size = FLAG.batch_size
save_every = FLAG.save_every
save_log = (FLAG.save_log == 1)
save_best_only = (FLAG.save_best_only == 1)

scale = FLAG.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3 or 4")

ckpt_dir = FLAG.ckpt_dir
if (ckpt_dir == "") or (ckpt_dir == "default"):
    ckpt_dir = f"checkpoint/x{scale}"
model_path = os.path.join(ckpt_dir, f"ESPCN-x{scale}.h5")


# -----------------------------------------------------------
#  Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"
lr_crop_size = 17
hr_crop_size = lr_crop_size * scale

samples = 150

train_set = dataset(dataset_dir, "train")
train_set.generate(lr_crop_size, hr_crop_size, samples)
train_set.load_data()

valid_set = dataset(dataset_dir, "valid")
valid_set.generate(lr_crop_size, hr_crop_size, samples)
valid_set.load_data()

# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

def main():
    model = ESPCN(scale)
    model.setup(optimizer=Adam(learning_rate=2e-4),
                loss=MeanSquaredError(),
                model_path=model_path,
                metric=PSNR)
    
    model.load_checkpoint(ckpt_dir)
    model.train(train_set, valid_set, steps=steps, batch_size=batch_size,
                save_best_only=save_best_only, save_every=save_every,
                save_log=save_log, log_dir=ckpt_dir)

if __name__ == "__main__":
    main()
