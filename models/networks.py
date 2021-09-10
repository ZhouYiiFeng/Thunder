import torch
import logging
from models.modules.thunder import  *
from models.modules.Subnet_constructor import subnet
import math
logger = logging.getLogger('base')


####################s
# define network
####################
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    netG = Network(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num)
    # netG = InvNet(channel_in=opt_net['in_nc'], channel_out=opt_net['out_nc'],
    #               subnet_constructor=subnet(subnet_type, init), block_num=opt_net['block_num'], down_num=down_num)
    return netG

def define_subG(opt):
    opt_net = opt['network_subG']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    netG = Network(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['block_num'], down_num)

    return netG
