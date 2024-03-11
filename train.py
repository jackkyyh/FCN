import torch

from loader import get_loader
from models import FCN8s_voc, Config
from trainer import Trainer

config = Config()
config.model_name = 'fcn8s_hybrid'
config.data="HYBRID"
config.data_path = 'datasets/benchmark_RELEASE/'
config.val_data_path = 'datasets/VOCdevkit'
config.lr  = 1e-4
config.n_epoch = 200
config.batch_size = 2

train_loader = get_loader(config,train=True)
test_loader = get_loader(config,train=False)
model = torch.nn.DataParallel(FCN8s_voc(config)) 

trainer = Trainer(model, train_loader, test_loader, config)
trainer.train()