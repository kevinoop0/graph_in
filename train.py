import os
import argparse
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
from model import Model
import visdom
import fire
import ipdb
import numpy as np


class Config(object):
    epoch = 20
    lr = 5e-5
    batch_size = 64
    gpu_id = 'cuda:0'
    loss_interval = 20
    img_interval = 50
    test_bs = 3


def train(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = torch.device(opt.gpu_id)
    vis = visdom.Visdom(env='train')  # python -m visdom.server

    train_dataset = PreprocessDataset(
        '/data/lzd/train_data/content', '/data/lzd/train_data/style')
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True)

    test_dataset = PreprocessDataset(
        '/data/lzd/test_data/content', '/data/lzd/test_data/style')
    test_loader = DataLoader(
        test_dataset, batch_size=opt.test_bs, shuffle=False)
    test_iter = iter(test_loader)

    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')
    model = Model()
    model= torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(device)
    optimizer = Adam(model.parameters(), lr=opt.lr)

    for e in range(1, opt.epoch):
        print(f'start {e} epoch:')
        for i, (content, style) in enumerate(train_loader, 1):
            content = content.to(device)  # [8, 3, 256, 256]
            style = style.to(device)  # [8, 3, 256, 256]
            loss,_ = model(content, style)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            print(
                f'[{e}/{opt.epoch} epoch],[{(e-1)*iters+i} /'f'{round(iters/opt.batch_size)}]: {loss.mean().item()}')

            if i % opt.loss_interval == 0:
                vis.line(Y=np.array([loss.mean().item()]), X=np.array(
                    [(e-1)*iters+i]), win='loss', update='append')

            if i % opt.img_interval == 0:
                c, s = next(test_iter)
                c = c.to(device)
                s = s.to(device)
                with torch.no_grad():
                    _, out = model(c, s)
                c = denorm(c, device)
                s = denorm(s, device)
                out = denorm(out, device)
                res = torch.cat([c, s, out], dim=0)
                vis.images(torch.clamp(res, 0, 1), win='image', nrow=opt.test_bs)

        # ipdb.set_trace()


if __name__ == '__main__':
    fire.Fire()
