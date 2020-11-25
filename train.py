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
    epoch = 15
    lr = 5e-5
    batch_size = 16
    gpu_id = 'cuda:0'
    loss_interval = 10
    img_interval = 500
    test_bs = 3


def train(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = torch.device(opt.gpu_id)
    vis = visdom.Visdom(env='gin')  # python -m visdom.server

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
    # model = Model()
    # model= torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    # model.to(device)
    model = Model().to(device)
    optimizer = Adam([
        {'params': model.decoder.parameters(), 'lr': opt.lr},
        {'params': model.gcn.parameters(), 'lr': opt.lr*10}], lr=opt.lr)

    for e in range(1, opt.epoch):
        print(f'start {e} epoch:')
        for i, (content, style) in enumerate(train_loader, 1):
            content = content.to(device)  # [8, 3, 256, 256]
            style = style.to(device)  # [8, 3, 256, 256]
            loss_c, loss_s, _ = model(content, style)
            loss = loss_c + 10 * loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f'[{e}/{opt.epoch} epoch],[{i} /'f'{round(iters/opt.batch_size)}]: {loss_c.item()} and {loss_s.item()}')

            if i % opt.loss_interval == 0:
                vis.line(Y=np.array([loss_c.item()]), X=np.array(
                    [(e-1)*round(iters/opt.batch_size)+i]), win='loss_c', update='append', opts=dict(xlabel='iteration',
                                                                                                     ylabel='Content loss',
                                                                                                     title='loss_c',
                                                                                                     legend=['Loss']))
                vis.line(Y=np.array([loss_s.item()]), X=np.array(
                    [(e-1)*round(iters/opt.batch_size)+i]), win='loss_s', update='append', opts=dict(xlabel='iteration',
                                                                                                     ylabel='style loss',
                                                                                                     title='loss_s',
                                                                                                     legend=['Loss']))
                vis.line(Y=np.array([loss.item()]), X=np.array(
                    [(e-1)*round(iters/opt.batch_size)+i]), win='loss', update='append', opts=dict(xlabel='iteration',
                                                                                                     ylabel='Total loss',
                                                                                                     title='loss',
                                                                                                     legend=['Loss']))

            if i % opt.img_interval == 0:
                c, s = next(test_iter)
                c = c.to(device)
                s = s.to(device)
                with torch.no_grad():
                    out = model.generate(c, s)
                c = denorm(c, device)
                s = denorm(s, device)
                out = denorm(out, device)
                res = torch.cat([c, s, out], dim=0)
                vis.images(torch.clamp(res, 0, 1),
                           win='image', nrow=opt.test_bs)


if __name__ == '__main__':
    fire.Fire()
