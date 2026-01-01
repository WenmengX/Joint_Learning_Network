"""
    Function:   Define some optional arguments and configurations
"""

import argparse
import time
import os


class opt():
    def __init__(self):
        time_stamp = time.time()
        local_time = time.localtime(time_stamp)
        self.time = time.strftime('%m%d%H%M', local_time)

    def parse(self):
        """ Function: Define optional arguments
        """
        parser = argparse.ArgumentParser(description='Self-supervised learing for multi-channel audio processing')

        # for both training and test stages
        parser.add_argument('--gpu-id', type=str, default='0,1', metavar='GPU', help='GPU ID (default: 7)')
        parser.add_argument('--workers', type=int, default=0, metavar='Worker', help='number of workers (default: 0)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training (default: False)')
        parser.add_argument('--use-amp', action='store_true', default=False,
                            help='Use automatic mixed precision training (default: False)')
        parser.add_argument('--seed', type=int, default=1, metavar='Seed', help='random seed (default: 1)')
        # parser.add_argument('--train', action='store_true', default=False, help='change to train stage (default: False)')
        parser.add_argument('--train', action='store_true', default=True,
                            help='change to train stage (default: False)')
        parser.add_argument('--test', action='store_true', default=False, help='change to test stage (default: False)')
        parser.add_argument('--dev', action='store_true', default=False, help='change to test stage (default: False)')
        parser.add_argument('--checkpoint-start', action='store_true', default=False,
                            help='train model from saved checkpoints (default: False)')
        parser.add_argument('--time', type=str, default=self.time, metavar='Time', help='time flag')

        parser.add_argument('--sources', type=int, nargs='+', default=[1], metavar='Sources',
                            help='Number of sources (default: 1, 2)')
        parser.add_argument('--source-state', type=str, default='mobile', metavar='SourceState',
                            help='State of sources (default: Mobile)')
        parser.add_argument('--localize-mode', type=str, nargs='+', default=['IDL', 'kNum', 1], metavar='LocalizeMode',
                            help='Mode for localization (default: Iterative detection and localization method, Unknown source number, Maximum source number is 2)')
        # e.g., ['IDL','unkNum', 2], ['IDL','kNum', 1], ['PD','kNum', 1]

        # for training stage
        parser.add_argument('--bz', type=int, nargs='+', default=[1, 1, 1], metavar='TrainValTestBatch',
                            help='batch size for training, validation and test (default: 1, 1, 5)')
        # parser.add_argument('--val-bz', type=int, default=2 , metavar='ValBatch', help='batch size for validating (default: 2)')
        # parser.add_argument('--gpu-call-bz', type=int, default=1, metavar='GPUCallBatch', help='batch size of each gpu call (default: 2)')
        parser.add_argument('--epochs', type=int, default=100, metavar='Epoch',
                            help='number of epochs to train (default: 100)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default:0.001)')
        parser.add_argument('--datasetMode', type=str, default='locata', metavar='datasetMode',
                            help='select dataset (default:simulate)')

        # parser.add_argument('--data-random', action='store_true', default=False, help='random condition for training data (default: False)')

        args = parser.parse_args()
        self.time = args.time

        if (args.train + args.test + args.dev) != 1:
            raise Exception('Stage of train or test is unrecognized')

        return args

    def dir(self):
        """ Function: Get directories of code, data and experimental results
        """
        """
        work_dir = '/home/wenmeng/Desktop/Data/DNS challenge 3/dns3_snr-5_20_t60_0p2_1_real-world'
        dirs = {}
        dirs['data'] = work_dir + '/data'
        dirs['exp'] = work_dir + '/exp'

        # source data
        dirs['sousig_train'] = dirs['data'] + '/LibriSpeech/train-clean-100'
        dirs['sousig_test'] = dirs['data'] + '/LibriSpeech/test-clean'
        dirs['sousig_dev'] = dirs['data'] + '/LibriSpeech/dev-clean'
        # noise data
        dirs['noisig_train'] = dirs['data'] + '/NoiSig/Noise92'
        dirs['noisig_test'] = dirs['data'] + '/NoiSig/Noise92'
        dirs['noisig_dev'] = dirs['data'] + '/NoiSig/Noise92'
        # experimental data
        '''
        dirs['sensig_train'] = dirs['data'] + '/low_reverb_data' + '/train'
        dirs['sensig_test'] = dirs['data'] + '/low_reverb_data' + '/test'
        dirs['sensig_dev'] = dirs['data'] + '/low_reverb_data' + '/dev'
        '''
        """
        ''' # dns3 + vislab
        dirs = {}
        dirs_data_dns3 = '/hdd1/data_dnsmos/selected_dns3_14000_snr_-5_20_t60_0p2_1_maxOrder3_real_world_5pos_02pi'
        dirs['dns3_sensig_train'] = dirs_data_dns3 + '/train'
        dirs['dns3_sensig_test'] = dirs_data_dns3 + '/test'
        dirs['dns3_sensig_dev'] = dirs_data_dns3 + '/dev'
        dir_data_vislab = '/hdd1/data_dnsmos/selected_dns3_Vislab_snr_-5_20_t60_0.385s_real_world_5pos_02pi'
        dirs['vislab_sensig_train'] = dir_data_vislab + '/train'
        dirs['vislab_sensig_test'] = dir_data_vislab + '/test'
        dirs['vislab_sensig_dev'] = dir_data_vislab + '/dev'
        '''
        # for Librispeech
        dirs = {}
        dirs_data = '/hdd1/Wenmeng/Librispeech_snr-5_20_t60MaterialFac_0p2_1_MaxOrd3_real_world'
        dirs['sensig_train'] = dirs_data + '/train'
        dirs['sensig_test'] = dirs_data + '/test'
        dirs['sensig_dev'] = dirs_data + '/dev'


        log_dir = r'.'
        log_dir = os.path.abspath(os.path.expanduser(log_dir))
        dirs['log'] = log_dir + '/exp' + '/with real world noise' + '/Librispeech'
        #dirs['log'] = log_dir + '/exp' + '/with real world noise'  + '/DNS3+VISLab'

        return dirs


if __name__ == '__main__':
    opts = opt()
    args = opts().parse()
    dirs = opts().dir()
