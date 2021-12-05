import torch
import torch.nn as nn
from torch.cuda.amp import autocast


class AudioModel(nn.Module):