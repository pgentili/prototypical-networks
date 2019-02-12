import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        
        self.encoder = encoder

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        learnable_scale = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)
        if xq.is_cuda:
          learnable_scale = learnable_scale.cuda()

        dists = euclidean_dist(zq, z_proto) * learnable_scale

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

class ModularProtonet(Protonet):
    def __init__(self, module_lists):
        super(Protonet, self).__init__()
        self.module_lists = module_lists
        self.active_blocks = [0 for _ in module_lists]
        self.encoder = self.build_encoder(active_blocks)

    def build_encoder(self, blocks):
        return nn.Sequential(
            *[module_list[ix] for module_list, ix in zip(self.module_lists,
                                                         active_modules)],
            Flatten()
        )

    def propose_new(self):
        rand_list_ix = random.rand_range(len(self.module_lists))
        rand_module_ix = random.rand_range(len(self.module_lists[rand_list_ix]))
        self.old_block = (rand_list_ix, self.active_blocks[rand_list_ix])

        # Update with new block
        self.active_blocks[rand_list_ix] = rand_module_ix
        self.encoder = self.build_encoder(self.proposed_blocks)

    def reject_new(self):
        ix, block = self.old_block
        self.active_blocks[ix] = block
        self.encoder = self.old_encoder


@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)

@register_model('modular_protonet')
def load_modular_protonet(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    choices_per = 2
    module_lists = [
        [conv_block(x_dim[0], hid_dim) for _ in range(choices_per)],
        [conv_block(hid_dim, hid_dim) for _ in range(choices_per)],
        [conv_block(hid_dim, hid_dim) for _ in range(choices_per)],
        [conv_block(x_dim[0], z_dim) for _ in range(choices_per)]
    ]

    return ModularProtonet(module_lists)
