import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
parser.add_argument('--gpus', type=str, required=False, default='0,1,2,3')
parser.add_argument('--lr', type=float, required=False, default=3e-4)
parser.add_argument('--batch_size', type=int, required=False, default=400)
parser.add_argument('--n_class', type=int, required=False, default=500)
parser.add_argument('--num_workers', type=int, required=False, default=8)
parser.add_argument('--max_epoch', type=int, required=False, default=120)
parser.add_argument('--test', type=str2bool,  required=False, default=False)

# load opts
parser.add_argument('--weights', type=str, required=False, default=None)

# save prefix
parser.add_argument('--save_prefix', type=str, required=False, default='checkpoints/lrw1000-baseline/')

# dataset
parser.add_argument('--dataset', type=str,  required=False, default='lrw1000')
parser.add_argument('--border', type=str2bool,  required=False, default=False)
parser.add_argument('--mixup', type=str2bool,  required=False, default=False)
parser.add_argument('--label_smooth', type=str2bool,  required=False, default=False)
parser.add_argument('--se', type=str2bool,  required=False, default=False)

args = parser.parse_args()

def getArgs():
    return args
