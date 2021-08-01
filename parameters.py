import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device("cuda:0")

EMBEDDING_SIZE = 32
HIDDEN_SIZE = 512
NUMBER_OF_LSTM_LAYERS = 1
DROPOUT = 0.5

uniform_init = 0.1
learning_rate = 0.001
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 12

maxEpochs = 4
log_every = 10
test_every = 2000

max_patience = 5
max_trials = 5
