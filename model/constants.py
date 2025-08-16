RANDOM_SEED = 42

ROOT_PATH = '/Users/simhyeongju/AVAPT/'

CIPI_PATH = ROOT_PATH + 'data/CIPI/'
MIKROKOSMOS_PATH = ROOT_PATH + 'data/Mikrokosmos-difficulty/'

import torch
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'