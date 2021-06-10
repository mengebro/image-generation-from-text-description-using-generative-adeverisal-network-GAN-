import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from miscc.config import cfg



####################Loss for image2text ###################
def image_to_text_loss(output, target):
    # bs x T x vocab_size - > bs * T x vocab_size
    bs, T, vocab_size = output.shape
    output = output.view(-1, vocab_size)
    # bs x T -> bs * T
    target = target.view(-1)
    return F.cross_entropy(output, target)
