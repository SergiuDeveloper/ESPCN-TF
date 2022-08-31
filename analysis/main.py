import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from math import log10
import os


INPUT_DIR = 'input'
OUTPUT_DIR = 'output'


model_data = {}
model_names = []

for file in os.listdir(INPUT_DIR):
    if file.endswith('.txt'):
        input_file_path = os.path.join(INPUT_DIR, file)

        model_name = file.replace('.txt', '')
        model_names.append(model_name)

        model_data[model_name] = {
            'loss': [],
            'psnr': [],
            'ssim': [],
            'val_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'training_time': []
        }

        with open(input_file_path, 'r') as input_file:
            for input_file_line in input_file:
                loss, psnr, ssim, val_loss, val_psnr, val_ssim, _ = input_file_line.split()

                model_data[model_name]['loss'].append(float(loss))
                model_data[model_name]['psnr'].append(10 * log10(1 / float(loss)))
                model_data[model_name]['ssim'].append(float(ssim))
                model_data[model_name]['val_loss'].append(float(val_loss))
                model_data[model_name]['val_psnr'].append(10 * log10(1 / float(val_loss)))
                model_data[model_name]['val_ssim'].append(float(val_ssim))

for metric in ['loss', 'psnr', 'ssim', 'val_loss', 'val_psnr', 'val_ssim']:
    plt.clf()
    
    for model_name in model_names:
        y = model_data[model_name][metric][20:]
        plt.plot([i * 1000 for i in range(len(y))], y, label=model_name)
    plt.xlabel('step')
    plt.ylabel(metric)
    plt.legend()

    plt.savefig(f'{OUTPUT_DIR}/{metric}.png')
