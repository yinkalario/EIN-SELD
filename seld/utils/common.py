import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


def float_samples_to_int16(y):
  """Convert floating-point numpy array of audio samples to int16."""
  if not issubclass(y.dtype.type, np.floating):
    raise ValueError('input samples not floating-point')
  return (y * np.iinfo(np.int16).max).astype(np.int16)

  
def int16_samples_to_float32(y):
  """Convert int16 numpy array of audio samples to float32."""
  if y.dtype != np.int16:
    raise ValueError('input samples not int16')
  return y.astype(np.float32) / np.iinfo(np.int16).max


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except:
            self.handleError(record)  


def create_logging(logs_dir, filemode):
    """Create log objective.

    Args:
      logs_dir (Path obj): logs directory
      filenmode: open file mode
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    i1 = 0

    while logs_dir.joinpath('{:04d}.log'.format(i1)).is_file():
        i1 += 1
        
    logs_path = logs_dir.joinpath('{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        # format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=logs_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)
    logging.getLogger('').addHandler(TqdmLoggingHandler())

    dt_string = datetime.now().strftime('%a, %d %b %Y %H:%M:%S')
    logging.info(dt_string)
    logging.info('')

    return logging
  

def convert_ordinal(n):
    """Convert a number to a ordinal number

    """
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
    return ordinal(n)


def move_model_to_gpu(model, cuda):
    """Move model to GPU   

    """
    # TODO: change DataParallel to DistributedDataParallel
    model = torch.nn.DataParallel(model)
    if cuda:
        logging.info('Utilize GPUs for computation')
        logging.info('Number of GPU available: {}\n'.format(torch.cuda.device_count()))
        model.cuda()
    else:
        logging.info('Utilize CPU for computation')
    return model


def count_parameters(model):
    """Count model parameters

    """
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Total number of parameters: {}\n'.format(params_num))


def print_metrics(logging, writer, values_dict, it, set_type='train'):
    """Print losses and metrics, and write it to tensorboard

    Args:
      logging: logging
      writer: tensorboard writer
      values_dict: losses or metrics
      it: iter
      set_type: 'train' | 'valid' | 'test'
    """
    out_str = ''
    if set_type == 'train':
        out_str += 'Train: '
    elif set_type == 'valid':
        out_str += 'valid: '

    for key, value in values_dict.items():
        out_str += '{}: {:.3f},  '.format(key, value)
        writer.add_scalar('{}/{}'.format(set_type, key), value, it)
    logging.info(out_str)


