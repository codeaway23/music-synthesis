import random
import numpy as np
from torch import manual_seed, device


# torch params
device = device("cuda")

# set seed
manual_seed(8)
random.seed(8)
np.random.seed(8)
