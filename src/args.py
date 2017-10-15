from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Neural Event Extraction model (NNED)')
    parser.add_argument('--no_use_conv', action='store_false', help='not use conv', dest='use_conv')
    parser.add_argument('--no_use_pos', action='store_false', help='not use position in conv', dest='use_position')
    parser.add_argument('--no_use_bilstm', action='store_false', help='not use bilstm', dest='bilstm')
    parser.add_argument('--no_use_pretrain', action='store_false', help='not use pretrain embedding', dest='use_pretrain')
    parser.add_argument('--no_shuffle_train', action='store_false', help='not shuffle train before train each epoch', dest='shuffle_train')
    parser.add_argument('--test_as_dev', action='store_true', help='use test as dev to get upper bound')

    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='gpu')
    #parser.add_argument('--gpu_core', type=int, default=0, help='GPU device to use') # use -1 for CPU
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--conv_width1', type=int, default=2)
    parser.add_argument('--conv_width2', type=int, default=3)
    parser.add_argument('--conv_filter_num', type=int, default=100)
    parser.add_argument('--hidden_dim_snd', type=int, default=-1)
    parser.add_argument('--loss_flag', type=str, default='cross-entropy') # or use 'nlloss'
    parser.add_argument('--opti_flag', type=str, default='sgd') # or use 'adadelta', 'adam', 'sgd'

    parser.add_argument('-train', type=str, default='', help="train file path")
    parser.add_argument('-dev', type=str, default='', help="dev file path")
    parser.add_argument('-test', type=str, default='', help="test file path")
    parser.add_argument('-pretrain_embed', type=str, default='', help="pretrain embed path")
    parser.add_argument('-tag', type=str, default='', help="tagset file path")
    parser.add_argument('-vocab', type=str, default='', help="vocab file path")
    parser.add_argument('-model', type=str, default='', help="model path")

    args = parser.parse_args()
    return args

