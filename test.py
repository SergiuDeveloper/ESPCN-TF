import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.common import *
from model import ESPCN
from tensorflow.keras.losses import MSE
from datetime import datetime, timedelta
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',     type=int, default=4,  help='-')
parser.add_argument('--ckpt-path', type=str, default="", help='-')

# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/x{scale}/ESPCN-x{scale}.h5" 

sigma = 0.3 if scale == 2 else 0.2


# -----------------------------------------------------------
# test 
# -----------------------------------------------------------

def main():
    model = ESPCN(scale)
    model.load_weights(ckpt_path)

    ls_data = sorted_list("/kaggle/input/dataset-div2k/DIV2K_valid_LR/DIV2K_valid_LR")
    ls_labels = sorted_list("/kaggle/input/dataset-div2k/DIV2K_valid_HR/DIV2K_valid_HR")

    sum_mse = 0
    sum_psnr = 0
    sum_ssim = 0
    sum_runtime = timedelta()
    sum_gpu_usage = 0
    gpu_recordings = 0
    for i in range(0, len(ls_data)):
        lr_image = read_image(ls_data[i])
        lr_image = gaussian_blur(lr_image, sigma=sigma)
        hr_image = read_image(ls_labels[i])

        lr_image = rgb2ycbcr(lr_image)
        hr_image = rgb2ycbcr(hr_image)

        lr_image = norm01(lr_image)
        hr_image = norm01(hr_image)

        lr_image = tf.expand_dims(lr_image, axis=0)

        timestamp_before = datetime.now()
        cpu_usage_before = None
        gpu_usage_before = None
        cpu_usage_after = None
        gpu_usage_after = None
        try:
            gpu_usage_before = tf.config.experimental.get_memory_info("GPU:0")["current"]
        except:
            pass
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except:
            pass

        sr_image = model.predict(lr_image)[0]

        timestamp_after = datetime.now()
        try:
            gpu_usage_after = tf.config.experimental.get_memory_info("GPU:0")["peak"]
        except:
            pass

        sum_mse += tf.reduce_mean(MSE(hr_image, sr_image).numpy())
        sum_psnr += PSNR(hr_image, sr_image).numpy()
        sum_ssim += SSIM(hr_image, sr_image).numpy()
        sum_runtime += timestamp_after - timestamp_before
        if gpu_usage_before is not None and gpu_usage_after is not None:
            sum_gpu_usage += gpu_usage_after - gpu_usage_before
            gpu_recordings += 1

    trainable_params_count = np.sum([np.prod(v.get_shape().as_list()) for v in model.get_trainable_params()])
    print(f"Trainable Params: {trainable_params_count}")
    print(f"MSE: {sum_mse / len(ls_data)}")
    print(f"PSNR: {sum_psnr / len(ls_data)}")
    print(f"SSIM: {sum_ssim / len(ls_data)}")
    print(f"Runtime: {sum_runtime / len(ls_data)}")
    if gpu_recordings > 0:
        print(f"GPU Usage: {sum_gpu_usage / gpu_recordings} bytes")

if __name__ == "__main__":
    main()
