
import os
from Opt import opt

opts = opt()
args = opts.parse()
dirs = opts.dir()
 
os.environ["OMP_NUM_THREADS"] = str(8) # limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

import numpy as np
import torch
import time
import scipy.io

from tensorboardX import SummaryWriter
import Dataset as at_dataset
import Learner as at_learner
import Model_LSTMbasedInter1 as at_model
import Module as at_module
from Dataset import Parameter
from utils import set_seed, set_random_seed
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	set_seed(42)
	speed = 343.0
	fs = 16000
	T = 4.79 # Trajectory length (s) 
	array_setup = at_dataset.dualch_array_setup
	array_locata_name = 'dicit'
	win_len = 512
	nfft = 512
	win_shift_ratio = 0.5
	fre_used_ratio = 1

	seg_fra_ratio = 12 # one estimate per segment (namely seg_fra_ratio frames) 
	seg_len = int(win_len*win_shift_ratio*(seg_fra_ratio+1))
	seg_shift = int(win_len*win_shift_ratio*seg_fra_ratio)

	segmenting = at_dataset.Segmenting_SRPDNN(K=seg_len, step=seg_shift, window=None)

	# Room acoustics for Librispeech
	dataset_train = at_dataset.FixTrajectoryDataset(
		data_dir=dirs['sensig_train'],
		#dataset_sz = 166816,
		dataset_sz=12000,
		transforms=[segmenting]
	)	
	dataset_dev = at_dataset.FixTrajectoryDataset(
		data_dir=dirs['sensig_dev'],
		dataset_sz = 1200,
		transforms=[segmenting]
	)
	dataset_test = at_dataset.FixTrajectoryDataset(
		data_dir=dirs['sensig_test'],
		dataset_sz=2000,
		transforms=[segmenting]
	)

	"""
	# Room acoustics for mixed data: DNS3 + Vislab
	dataset_dns3_train = at_dataset.FixTrajectoryDataset(
		data_dir=dirs['dns3_sensig_train'],
		dataset_sz=14000,
		transforms=[segmenting]
	)

	dataset_dns3_dev = at_dataset.FixTrajectoryDataset(
		data_dir=dirs['dns3_sensig_dev'],
		dataset_sz=2800,
		transforms=[segmenting]
	)
	dataset_vislab_train = at_dataset.FixTrajectoryDataset(
		data_dir=dirs['vislab_sensig_train'],
		dataset_sz=1000,
		transforms=[segmenting]
	)
	dataset_vislab_dev = at_dataset.FixTrajectoryDataset(
		data_dir=dirs['vislab_sensig_dev'],
		dataset_sz=200,
		transforms=[segmenting]
	)

	mixed_dataset_train = at_dataset.MixedDataset(dataset_dns3_train, dataset_vislab_train, ratio_1_to_2=14)
	mixed_dataset_dev = at_dataset.MixedDataset(dataset_dns3_dev, dataset_vislab_dev, ratio_1_to_2=14)
	"""
	# %% Network declaration, learner declaration
	tar_useVAD = True
	ch_mode = 'MM' 
	res_the = 37 # Maps resolution (elevation) 
	res_phi = 73 # Maps resolution (azimuth) 

	net = at_model.Joint_Learning()
	# from torchsummary import summary
	# summary(net,input_size=(4,256,100),batch_size=55,device="cpu")
	print('# Parameters:', sum(param.numel() for param in net.parameters())/1000000, 'M')

	learner = at_learner.SourceTrackingFromSTFTLearner(net, win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft, fre_used_ratio=fre_used_ratio,
				nele=res_the, nazi=res_phi, rn=array_setup.mic_pos, fs=fs, ch_mode = ch_mode, tar_useVAD = tar_useVAD, localize_mode = args.localize_mode) 

	if len(args.gpu_id)>1:
		learner.mul_gpu()  # commented by Wenmeng
	if use_cuda:
		learner.cuda()
	else:
		learner.cpu()
	if args.use_amp:
		learner.amp()
	kwargs = {'num_workers': args.workers, 'pin_memory': True}  if use_cuda else {}

	gamma = 0.96
	print('Training Stage!')

	if args.checkpoint_start:
		learner.resume_checkpoint(checkpoints_dir=dirs['log'], from_latest=True) # Train from latest checkpoints

		# %% TensorboardX
	train_writer = SummaryWriter(dirs['log'] + '/train/', 'train')
	val_writer = SummaryWriter(dirs['log'] + '/val/', 'val')

	lr0 = 0.0005
	nepoch = 200


	# delete num_workers in debug mode
	#dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.bz[0], shuffle=True)
	batch_size = 1
	# for Librispeech
	dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
	#dataloader_val = torch.utils.data.DataLoader(dataset=dataset_dev, batch_size=args.bz[1], shuffle=False)
	dataloader_val = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.bz[1], shuffle=False)

	''' # for dns3 or vislab
	dataloader_train = torch.utils.data.DataLoader(dataset=dataset_dns3_train, batch_size=batch_size, shuffle=True)
	dataloader_val = torch.utils.data.DataLoader(dataset=dataset_dns3_dev, batch_size=args.bz[1], shuffle=False)
	'''
	#learner.model.load_state_dict(torch.load("/home/wenmeng/Desktop/codes/test11_joint_asymmetric_spatialnet/exp/with real world noise/Librispeech/best_model.tar")['model'])
	learner.start_epoch = 1
	lr = lr0 * math.pow(gamma,learner.start_epoch-1)
	for epoch in range(learner.start_epoch, nepoch+1, 1):
		print('\nEpoch {}/{}:'.format(epoch, nepoch))
		lr = lr0 * math.pow(gamma, epoch-1)
		if lr < 0.00001:
			lr = 0.00001
		print(lr)
		loss_x_train = learner.train_epoch(dataloader_train, batch_size, len_each_epoch=12000, lr=lr, epoch=epoch, return_metric=False)  # len_each_epoch= 500

		loss_x_val, metric_val = learner.test_epoch(dataloader_val, batch_size, epoch=epoch, return_metric=True)
		print('Test loss: {:.4f}, Test ACC: {:.2f}%, Test MAE: {:.2f}deg, Test RMSE: {:.2f}deg. '.format(loss_x_val, metric_val['ACC'] * 100, metric_val['MAE'], metric_val['rmse']))

			# Save model
		is_best_epoch = learner.is_best_epoch(current_score=loss_x_val*(-1))
		learner.save_checkpoint(epoch=epoch, checkpoints_dir=dirs['log'], is_best_epoch=is_best_epoch)



	print('\nTraining finished\n')
