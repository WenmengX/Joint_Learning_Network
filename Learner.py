import os
import gc
import numpy as np
import torch
import torch.optim as optim
import webrtcvad
import pmsqe_torch
from copy import deepcopy
from abc import ABC, abstractmethod
from tqdm import tqdm, trange
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from metrics import SI_SDR, STOI, NB_PESQ, WB_PESQ, PY_WB_PESQ, PY_NB_PESQ
from utils import sph2cart, cart2sph,forgetting_norm
from torch.nn.utils import	clip_grad_norm_
import Module as at_module
from stoi_loss import stoi_loss
import matplotlib.pyplot as plt
from utils import mc_stft, cv_istft
import math

class Learner(ABC):
    """ Abstract class to the routines to train the one source tracking models and perform inferences.
    """
    def __init__(self, model):
        self.model = model
        # self.cuda_activated = False
        self.max_score = -np.inf
        self.use_amp = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.start_epoch = 1
        #self.device = device
        super().__init__()

    def mul_gpu(self):
        self.model = torch.nn.DataParallel(self.model)  # commented by Wenmeng
        #self.model = torch.nn.DataParallel(self.model, device_ids=[1])  # commented by Wenmeng
    # When multiple gpus are used, 'module.' is added to the name of model parameters.
    # So whether using one gpu or multiple gpus should be consistent for model traning and checkpoints loading.

    def cuda(self):
        """ Move the model to the GPU and perform the training and inference there.
        """
        torch.cuda.set_device(0)
        #torch.cuda.set_device(1)
        self.model.cuda()
        self.device = "cuda:0"
    # self.cuda_activated = True

    def cpu(self):
        """ Move the model back to the CPU and perform the training and inference here.
        """
        self.model.cpu()
        self.device = "cpu"
        # self.cuda_activated = False

    def amp(self):
        """ Use Automatic Mixed Precision to train network.
        """
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @abstractmethod
    def data_preprocess(self, mic_sig_batch=None, dp_mic_sig_batch=None, acoustic_scene_batch=None, vad_batch=None):
        """ To be implemented in each learner according to input of their models
        """
        pass

    @abstractmethod
    def predgt2DOA(self, pred_batch=None, gt_batch=None):
        """
        """
        pass

    def ce_loss(self, pred_batch, gt_batch):
        """ To be implemented in each learner according to output of their models
        """
        pass

    @abstractmethod
    def mse_loss(self, pred_batch, gt_batch):
        """ To be implemented in each learner according to output of their models
        """
        pass

    @abstractmethod
    def evaluate(self, pred, gt):
        """ To be implemented in each learner according to output of their models
        """
        pass

    def train_epoch(self, dataset, batch_size, len_each_epoch=500, lr=0.0001, epoch=None, return_metric=False):
        """ Train the model with an epoch of the dataset.
        """
        loss_se_total= 0
        loss_doa_total= 0
        avg_loss = 0
        avg_beta = 0.99
        self.use_amp = True
        # self.model.load_state_dict(torch.load("/home/wenmeng/Desktop/codes/joint_loc_se_LSTM_v4_para_model2_FFinter_SEmetric/exp/with real world noise/best_model.tar"))
        self.model.train()
        #lr = 0
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        loss = 0

        loss_train = 0
        if return_metric:
            metric = {}

        optimizer.zero_grad()
        #pbar = tqdm(enumerate(dataset), total=len(dataset), leave=False)

        pbar = trange(len_each_epoch//batch_size, ascii=True)
        # for gpu_call_idx in pbar:

        dataS = np.zeros((batch_size,dataset.dataset[0][0].shape[0],dataset.dataset[1][0].shape[1]))
        dp_dataS = np.zeros(dataS.shape)
        gtD = np.zeros((batch_size, dataset.dataset[0][2]['doa'].shape[0], dataset.dataset[0][2]['doa'].shape[1],dataset.dataset[0][2]['doa'].shape[2]))
        gtV = np.zeros((batch_size, dataset.dataset[0][2]['vad_sources'].shape[0], dataset.dataset[0][2]['vad_sources'].shape[1],
                        dataset.dataset[0][2]['vad_sources'].shape[2]))

        #for batch_idx, (mic_sig_batch, gt_batch) in pbar:
        for batch_idx in pbar:
            if epoch is not None: pbar.set_description('Epoch {}'.format(epoch))

            for i in range(batch_size):
                dataS[i, :, :] = np.expand_dims(dataset.dataset[(batch_idx - 1) * batch_size + i][0], axis=0)
                dp_dataS[i, :, :] = np.expand_dims(dataset.dataset[(batch_idx - 1) * batch_size + i][1], axis=0)
                gtD[i, :, :, :] = dataset.dataset[(batch_idx - 1) * batch_size + i][2]['doa']
                gtV[i, :, :, :] = dataset.dataset[(batch_idx - 1) * batch_size + i][2]['vad_sources']
            mic_sig_batch = torch.from_numpy(dataS)
            dp_mic_sig_batch = torch.from_numpy(dp_dataS)
            gtDT = torch.from_numpy(gtD)
            gtVT = torch.from_numpy(gtV)
            gt_batch = {'doa':gtDT, 'vad_sources':gtVT}

            dp_in_batch, in_batch, dp_mag, mag, dp_phase, phase,  dp_stft_rebatch, stft_rebatch, gt_batch, data_RIRI, dp_data_RIRI = self.data_preprocess(mic_sig_batch, dp_mic_sig_batch, gt_batch)

            data = torch.cat((mag[0], phase[0]), 1)
            dp_stft_re = torch.cat((dp_stft_rebatch[:, 0, :, :], dp_stft_rebatch[:, 1, :, :]), dim=-1).permute(0, 2,1)
            dp_stft_re = torch.abs(dp_stft_re).to(torch.float32)

            data.requires_grad_()
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                data_x = (data, data_RIRI)
                pred_batch2, x_dereverb = self.model(data_x)

                loss_doa = self.mse_loss(pred_batch=pred_batch2, gt_batch=gt_batch)

            x_dereverb_1 = torch.view_as_complex(torch.stack((x_dereverb[..., 0], x_dereverb[..., 1]), dim=-1))
            x_dereverb_2 = torch.view_as_complex(torch.stack((x_dereverb[..., 2], x_dereverb[..., 3]), dim=-1))
            enhanced_istft_1 = cv_istft(x_dereverb_1,  n_fft=self.nfft, hop_length= math.floor(self.win_shift_ratio * self.win_len), win_length=self.win_len,
                                        length=dataS.shape[1])
            enhanced_istft_2 = cv_istft(x_dereverb_2,  n_fft=self.nfft, hop_length= math.floor(self.win_shift_ratio * self.win_len), win_length=self.win_len,
                                        length=dataS.shape[1])
            dp_mic_sig_batch_1 = dp_mic_sig_batch[..., 0].to(self.device)
            dp_mic_sig_batch_2 = dp_mic_sig_batch[..., 1].to(self.device)
            loss_se_1 = -si_sdr(preds=enhanced_istft_1, target=dp_mic_sig_batch_1)
            loss_se_2 = -si_sdr(preds=enhanced_istft_2, target=dp_mic_sig_batch_2)
            loss_se = (loss_se_1 + loss_se_2) / 2  # + (pmsqe_loss + mse_loss_pesq)*0.05

            loss_x = loss_se + loss_doa

            # add up gradients until optimizer.zero_grad(), multiply a scale to gurantee the gradients equal to that when trajectories_per_gpu_call = trajectories_per_batch
            if self.use_amp:
                self.scaler.scale(loss_x).backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss_x.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()

            avg_loss = avg_beta * avg_loss + (1 - avg_beta) * loss_x.item()
            pbar.set_postfix(loss=avg_loss / (1 - avg_beta ** (batch_idx + 1)))
            # pbar.set_postfix(loss=loss.item())
            pbar.update()

            loss_train += loss_x.item()
            loss_doa_total += loss_doa.item()
            loss_se_total += loss_se.item()
            gc.collect()
        loss_train /= len(pbar)
        loss_doa_total /= len(pbar)
        loss_se_total /= len(pbar)

        print('loss_train:{:.4f},loss_doa: {:.4f}, loss_se: {:.4f}'.format(loss_train, loss_doa_total, loss_se_total ))

        return loss_train #loss_doa, loss_se

    def test_epoch(self, dataset, batch_size, epoch= None,return_metric=False):
        """ Test the model with an epoch of the dataset.
        """

        self.model.eval()
        return_metric = True
        with torch.no_grad():
            loss_test_x = 0
            total_stoi_score = 0
            total_wb_pesq_score = 0
            total_wb_pypesq_score = 0
            total_nb_pesq_score = 0
            total_nb_pypesq_score = 0
            total_SI_snr_score = 0
            total_SI_snr_score_noisy = 0
            total_SI_snr_score_new = 0
            idx = 0
            NoUtterCnt = 0
            if return_metric:
                metric = {}

            len_each_epoch = len(dataset.dataset)
            pbar = trange(len_each_epoch // batch_size, ascii=True)

            dataS = np.zeros((batch_size, dataset.dataset[0][0].shape[0], dataset.dataset[1][0].shape[1]))
            dp_dataS = np.zeros(dataS.shape)
            gtD = np.zeros((batch_size, dataset.dataset[0][2]['doa'].shape[0], dataset.dataset[0][2]['doa'].shape[1],
                            dataset.dataset[0][2]['doa'].shape[2]))
            gtV = np.zeros(
                (batch_size, dataset.dataset[0][2]['vad_sources'].shape[0],
                 dataset.dataset[0][2]['vad_sources'].shape[1],
                 dataset.dataset[0][2]['vad_sources'].shape[2]))

            for batch_idx in pbar:
                if epoch is not None: pbar.set_description('Epoch {}'.format(epoch))

                for i in range(batch_size):
                    dataS[i, :, :] = np.expand_dims(dataset.dataset[(batch_idx - 1) * batch_size + i][0], axis=0)
                    dp_dataS[i, :, :] = np.expand_dims(dataset.dataset[(batch_idx - 1) * batch_size + i][1], axis=0)
                    gtD[i, :, :, :] = dataset.dataset[(batch_idx - 1) * batch_size + i][2]['doa']
                    gtV[i, :, :, :] = dataset.dataset[(batch_idx - 1) * batch_size + i][2]['vad_sources']
                mic_sig_batch = torch.from_numpy(dataS)
                dp_mic_sig_batch = torch.from_numpy(dp_dataS)
                gtDT = torch.from_numpy(gtD)
                gtVT = torch.from_numpy(gtV)
                gt_batch = {'doa': gtDT, 'vad_sources': gtVT}
                dp_in_batch, in_batch, dp_mag, mag, dp_phase, phase,dp_stft_rebatch, stft_rebatch, gt_batch, data_RIRI, dp_data_RIRI = self.data_preprocess(mic_sig_batch, dp_mic_sig_batch, gt_batch)

                data = torch.cat((mag[0], phase[0]), 1)

                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    data_x = (data, data_RIRI)
                    pred_batch2, x_dereverb = self.model(data_x)
                    loss_doa = self.mse_loss(pred_batch=pred_batch2, gt_batch=gt_batch)

                x_dereverb_1 = torch.view_as_complex(torch.stack((x_dereverb[..., 0], x_dereverb[..., 1]), dim=-1))
                x_dereverb_2 = torch.view_as_complex(torch.stack((x_dereverb[..., 2], x_dereverb[..., 3]), dim=-1))
                enhanced_istft_1 = cv_istft(x_dereverb_1, n_fft=self.n_fft, hop_length= math.floor(self.win_shift_ratio * self.win_len), win_length=self.win_len,
                                            length=dataS.shape[1])
                enhanced_istft_2 = cv_istft(x_dereverb_2, n_fft=self.n_fft, hop_length= math.floor(self.win_shift_ratio * self.win_len), win_length=self.win_len,
                                            length=dataS.shape[1])
                enhanced_1 = enhanced_istft_1.detach().squeeze(0).cpu().numpy()
                enhanced_2 = enhanced_istft_2.detach().squeeze(0).cpu().numpy()
                dp_mic_sig_batch_1 = dp_mic_sig_batch[..., 0].to(self.device)
                dp_mic_sig_batch_2 = dp_mic_sig_batch[..., 1].to(self.device)
                clean_1 = dp_mic_sig_batch_1.detach().squeeze(0).cpu().numpy()
                clean_2 = dp_mic_sig_batch_2.detach().squeeze(0).cpu().numpy()

                if np.linalg.norm(clean_1) == 0:
                    NoUtterCnt += 1
                    continue

                mic_sig_batch_1 = mic_sig_batch[..., 0].to(self.device)
                mic_sig_batch_2 = mic_sig_batch[..., 1].to(self.device)
                noisy_1 = mic_sig_batch_1.detach().squeeze(0).cpu().numpy()
                noisy_2 = mic_sig_batch_2.detach().squeeze(0).cpu().numpy()

                loss_se_1 = -si_sdr(preds=enhanced_istft_1, target=dp_mic_sig_batch_1)
                loss_se_2 = -si_sdr(preds=enhanced_istft_2, target=dp_mic_sig_batch_2)

                loss_se = (loss_se_1 + loss_se_2) / 2    # + (pmsqe_loss + mse_loss_pesq) * 0.05
                loss_x = loss_doa +loss_se

                loss_test_x += loss_x.item()

                if return_metric:
                    stoi_score_sample = STOI(clean_1, enhanced_1, sr=self.fs)
                    wb_pesq_score_sample = WB_PESQ(clean_1, enhanced_1)
                    wb_pypesq_score_sample = PY_WB_PESQ(clean_1, enhanced_1)
                    nb_pesq_score_sample = NB_PESQ(clean_1, enhanced_1)
                    nb_pypesq_score_sample = PY_NB_PESQ(clean_1, enhanced_1)
                    si_sdr_score_sample_2 = SI_SDR(clean_2, enhanced_2, sr=self.fs)
                    si_sdr_score_sample_1 = SI_SDR(clean_1, enhanced_1, sr=self.fs)
                    si_sdr_score_sample_1_new = (0 - loss_se_1).item()
                    si_sdr_score_sample_2_new = (0 - loss_se_2).item()
                    si_sdr_score_noisy_1 = SI_SDR(clean_1, noisy_1, sr=self.fs)
                    si_sdr_score_noisy_2 = SI_SDR(clean_2, noisy_2, sr=self.fs)

                    pred_batch2, gt_batch = self.predgt2DOA(pred_batch=pred_batch2, gt_batch=gt_batch)
                    metric_batch = self.evaluate(pred=pred_batch2, gt=gt_batch)
                    if idx == 0:
                        for m in metric_batch.keys():
                            metric[m] = 0
                    for m in metric_batch.keys():
                        metric[m] += metric_batch[m].item()
                    idx = idx + 1

                    if wb_pesq_score_sample is not None:
                        total_wb_pesq_score += wb_pesq_score_sample
                        total_wb_pypesq_score += wb_pypesq_score_sample
                        total_nb_pesq_score += nb_pesq_score_sample
                        total_nb_pypesq_score += nb_pypesq_score_sample
                        total_stoi_score += stoi_score_sample
                        total_SI_snr_score_noisy += (si_sdr_score_noisy_1 + si_sdr_score_noisy_2) / 2
                        total_SI_snr_score += (si_sdr_score_sample_1 + si_sdr_score_sample_2) / 2
                        total_SI_snr_score_new += (si_sdr_score_sample_1_new + si_sdr_score_sample_2_new) / 2

                pbar.set_postfix(loss=loss_x.item())  # ACC = metric['ACC'], MAE = metric['MAE']
                pbar.update()

            if return_metric:
                for m in metric_batch.keys():
                    metric[m] /= len(pbar)
            loss_test_x /= len(pbar)

            total_wb_pesq_score /= (len(pbar) - NoUtterCnt)
            total_wb_pypesq_score /= (len(pbar) - NoUtterCnt)
            total_nb_pesq_score /= (len(pbar) - NoUtterCnt)
            total_nb_pypesq_score /= (len(pbar) - NoUtterCnt)
            total_stoi_score /= (len(pbar) - NoUtterCnt)
            total_SI_snr_score /= (len(pbar) - NoUtterCnt)
            total_SI_snr_score_noisy /= (len(pbar) - NoUtterCnt)
            total_SI_snr_score_new /= (len(pbar) - NoUtterCnt)

            print(
                'cv_loss: {:.4f},  wb_pesq: {:.4f},wb_pypesq:{:.4f}, nb_pesq: {:.4f}, nb_pypesq:{:.4f},stoi: {:.4f},si_snr: {:.4f},si_snr_noisy{:.4f} ,si_snr_new{:.4f}'.format(
                    loss_test_x, total_wb_pesq_score,total_wb_pypesq_score,total_nb_pesq_score , total_nb_pypesq_score, total_stoi_score, total_SI_snr_score, total_SI_snr_score_noisy, total_SI_snr_score_new))

            if return_metric:
                return loss_test_x, metric


    def predict_batch(self, gt_batch, mic_sig_batch, wDNN=True):
        """
        Function: Predict
        Args:
            mic_sig_batch
            gt_batch
        Returns:
            pred_batch		- [DOA, VAD] / [DOA]
            gt_batch		- [DOA, IPD, VAD] / [DOA, VAD]
            mic_sig_batch	- (nb, nsample, nch)
        """
        self.model.eval()
        with torch.no_grad():

            mic_sig_batch = mic_sig_batch.to(self.device)
            in_batch, gt_batch = self.data_preprocess_predict(mic_sig_batch, gt_batch)

            if wDNN:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    x_dereverb = self.model(in_batch)
                #pred_batch, gt_batch = self.predgt2DOA(pred_batch=pred_batch, gt_batch=gt_batch)
            else:
                nt_ori = in_batch.shape[-1]
                nt_pool = gt_batch['doa'].shape[1]
                time_pool_size = int(nt_ori/nt_pool)
                phase = in_batch[:, int(in_batch.shape[1]/2):, :, :].detach() # (nb*nmic_pair, 2, nf, nt)
                phased = phase[:,0,:,:] - phase[:,1,:,:]
                pred_batch = torch.cat((torch.cos(phased), torch.sin(phased)), dim=1).permute(0, 2, 1) # (nb*nmic_pair, nt, 2nf)
                pred_batch, gt_batch = self.predgt2DOA(pred_batch=pred_batch, gt_batch=gt_batch, time_pool_size=time_pool_size)

            return pred_batch, gt_batch, mic_sig_batch


    def predict(self, dataset, wDNN=True, return_predgt=False, metric_setting=None,save_file=False):
        """
        Function: Predict
        Args:
            metric_setting: ae_mode=ae_mode, ae_TH=ae_TH, useVAD=useVAD, vad_TH=vad_TH
        Returns:
            pred		- [DOA, VAD] / [DOA]
            gt			- [DOA, IPD, VAD] / [DOA, VAD]
            mic_sig		- (nb, nsample, nch)
            metric		- [ACC, MDR, FAR, MAE, RMSE]
        """
        data = []

        self.model.eval()
        with torch.no_grad():
            idx = 0
            if return_predgt:
                pred = []
                gt = []
                mic_sig = []
            if metric_setting is not None:
                metric = {}

            for mic_sig_batch, gt_batch in dataset:
                print('Dataloading: ' + str(idx+1))
                # print(mic_sig_batch.shape)
                mic_sig_batch = torch.cat((mic_sig_batch[:,:,8:9], mic_sig_batch[:,:,5:6]), axis=-1)
                pred_batch, gt_batch, mic_sig_batch = self.predict_batch(gt_batch, mic_sig_batch, wDNN)
                # print(mic_sig_batch.shape)

                if (metric_setting is not None):
                    if save_file:
                        metric_batch = self.evaluate(pred=pred_batch, gt=gt_batch,metric_setting=metric_setting, idx=idx)
                    else:
                        metric_batch = self.evaluate(pred=pred_batch, gt=gt_batch,metric_setting=metric_setting)
                if return_predgt:
                    pred += [pred_batch]
                    gt += [gt_batch]
                    mic_sig += [mic_sig_batch]
                if metric_setting is not None:
                    for m in metric_batch.keys():
                        if idx==0:
                            metric[m] = deepcopy(metric_batch[m])
                        else:
                            metric[m] = torch.cat((metric[m], metric_batch[m]), axis=0)

                idx = idx+1

            if return_predgt:
                data += [pred, gt]
                data += [mic_sig]
            if metric_setting is not None:
                data += [metric]
            return data

    def is_best_epoch(self, current_score):
        """ Check if the current model got the best metric score
        """
        if current_score >= self.max_score:
            self.max_score = current_score
            is_best_epoch = True
        else:
            is_best_epoch = False

        return is_best_epoch

    def save_checkpoint(self, epoch, checkpoints_dir, is_best_epoch = False):
        """ Save checkpoint to "checkpoints_dir" directory, which consists of:
            - the epoch number
            - the best metric score in history
            - the optimizer parameters
            - the model parameters
        """
        
        print(f"\t Saving {epoch} epoch model checkpoint...")
        if self.use_amp:
            state_dict = {
                "epoch": epoch,
                "max_score": self.max_score,
                # "optimizer": self.optimizer.state_dict(),
                "scalar": self.scaler.state_dict(),
                "model": self.model.state_dict()
            }
        else:
            state_dict = {
                "epoch": epoch,
                "max_score": self.max_score,
                # "optimizer": self.optimizer.state_dict(),
                "model": self.model.state_dict()
            }

        torch.save(state_dict, checkpoints_dir + "/latest_model.tar")
        torch.save(state_dict, checkpoints_dir + "/model"+str(epoch)+".tar")

        if is_best_epoch:
            print(f"\t Found a max score in the {epoch} epoch, saving...")
            torch.save(state_dict, checkpoints_dir + "/best_model.tar")


    def resume_checkpoint(self, checkpoints_dir, from_latest = True):
        """Resume from the latest/best checkpoint.
        """

        if from_latest:

            latest_model_path = checkpoints_dir + "/lightning.ckpt"

            assert os.path.exists(latest_model_path), f"{latest_model_path} does not exist, can not load latest checkpoint."

            # self.dist.barrier()  # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work

            # device = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            checkpoint = torch.load(latest_model_path, map_location=self.device)

            #self.start_epoch = checkpoint["epoch"] + 1
            #self.max_score = checkpoint["max_score"]
            # self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.use_amp:
                self.scaler.load_state_dict(checkpoint["scalar"])
            self.model.load_state_dict(checkpoint["state_dict"])

            # if self.rank == 0:
            print(f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

        else:
            best_model_path = checkpoints_dir + "/best_model.tar"

            assert os.path.exists(best_model_path), f"{best_model_path} does not exist, can not load best model."

            # self.dist.barrier()  # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work

            # device = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            checkpoint = torch.load(best_model_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model"])




class SourceTrackingFromSTFTLearner(Learner):
    """ Learner for models which use STFTs of multiple channels as input
    """
    def __init__(self, model, win_len, win_shift_ratio, nfft, fre_used_ratio, nele, nazi, rn, fs, ch_mode, tar_useVAD, localize_mode, c=343.0): #, arrayType='planar', cat_maxCoor=False, apply_vad=False):
        """
        fre_used_ratio - the ratio between used frequency and valid frequency
        """
        super().__init__(model)

        self.nele = nele
        self.nazi = nazi

        self.nfft = nfft
        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.fs = fs
        #self.nf_used = int(self.nfft/2*fre_used_ratio)
        if fre_used_ratio == 1:
            self.fre_range_used = range(1, int(self.nfft/2*fre_used_ratio)+1, 1)
        elif fre_used_ratio == 0.5:
            self.fre_range_used = range(0, int(self.nfft/2*fre_used_ratio), 1)
        else:
            raise Exception('Prameter fre_used_ratio unexpected')

        # self.nf_used = int((self.nfft / 2 +1)* fre_used_ratio)
        self.dostft = at_module.STFT(win_len=win_len, win_shift_ratio=win_shift_ratio, nfft=nfft)
        fre_max = fs / 2
        self.ch_mode = ch_mode
        self.gerdpipd = at_module.DPIPD(ndoa_candidate=[nele, nazi], mic_location=rn, nf=int(self.nfft/2) + 1, fre_max=fre_max,
                                        ch_mode=self.ch_mode, speed=c)
        self.tar_useVAD = tar_useVAD
        self.addbatch = at_module.AddChToBatch(ch_mode=self.ch_mode)
        self.removebatch = at_module.RemoveChFromBatch(ch_mode=self.ch_mode)
        self.sourcelocalize = at_module.SourceDetectLocalize(max_num_sources=int(localize_mode[2]), source_num_mode=localize_mode[1], meth_mode=localize_mode[0])

        self.getmetric = at_module.getMetric(source_mode='single')

    def data_preprocess(self, mic_sig_batch=None, dp_mic_sig_batch=None, gt_batch=None, vad_batch=None, eps=1e-6, nor_flag=True):

        data = []
        data_mag = []
        data_phase = []
        dp_data = []
        dp_data_mag = []
        dp_data_phase = []
        if mic_sig_batch is not None:
            mic_sig_batch = mic_sig_batch.to(self.device)
            dp_mic_sig_batch = dp_mic_sig_batch.to(self.device)
            stft = mc_stft(mic_sig_batch.permute(0, 2, 1), n_fft=self.nfft, hop_length= math.floor(self.win_shift_ratio * self.win_len), win_length=self.win_len)

            dp_stft = mc_stft(dp_mic_sig_batch.permute(0, 2, 1),  n_fft=self.nfft, hop_length= math.floor(self.win_shift_ratio * self.win_len), win_length=self.win_len)


            # change batch (nb,nch,nf,nt)→(nb*(nch-1),2,nf,nt)/(nb*(nch-1)*nch/2,2,nf,nt)
            stft_rebatch = self.addbatch(stft)
            dp_stft_rebatch = self.addbatch(dp_stft)
            if nor_flag:
                nb, nc, nf, nt = stft_rebatch.shape
                mag = torch.abs(stft_rebatch)
                phase = torch.angle(stft_rebatch)
                mean_value = forgetting_norm(mag)
                stft_rebatch_real = torch.real(stft_rebatch) / (mean_value + eps)
                stft_rebatch_image = torch.imag(stft_rebatch) / (mean_value + eps)
                mag = mag / (mean_value + eps)
                #phase = phase / (forgetting_norm(phase) + eps)

                dp_mag = torch.abs(dp_stft_rebatch)
                dp_mean_value = forgetting_norm(dp_mag)
                dp_phase = torch.angle(dp_stft_rebatch)
                dp_stft_rebatch_real = torch.real(dp_stft_rebatch) / (dp_mean_value + eps)
                dp_stft_rebatch_image = torch.imag(dp_stft_rebatch) / (dp_mean_value + eps)
                dp_mag = dp_mag / (dp_mean_value + eps)
                #dp_phase = dp_phase / (forgetting_norm(dp_phase) + eps)
            else:
                stft_rebatch_real = torch.real(stft_rebatch)
                stft_rebatch_image = torch.imag(stft_rebatch)

                dp_stft_rebatch_real = torch.real(dp_stft_rebatch)
                dp_stft_rebatch_image = torch.imag(dp_stft_rebatch)
            # prepare model input
            real_image_batch  =  torch.cat((stft_rebatch_real,stft_rebatch_image),dim=1)
            data += [real_image_batch]
            data_mag += [mag]
            data_phase += [phase]
            data_RIRI = torch.view_as_real(stft_rebatch.permute(0, 2, 3, 1)).reshape(nb, nf, nt, -1)  # B,F,T,2C  ReImReIm

            dp_real_image_batch = torch.cat((dp_stft_rebatch_real, dp_stft_rebatch_image), dim=1)
            dp_data += [dp_real_image_batch]
            dp_data_mag += [dp_mag]
            dp_data_phase += [dp_phase]
            dp_data_RIRI = torch.view_as_real(dp_stft_rebatch.permute(0, 2, 3, 1)).reshape(nb, nf, nt, -1)
            dp_stft_rebatch_norm = dp_stft_rebatch
            stft_rebatch_norm = stft_rebatch

        if gt_batch is not None:
            DOAw_batch = gt_batch['doa']
            vad_batch = gt_batch['vad_sources']

            source_doa = DOAw_batch.cpu().numpy()

            if self.ch_mode == 'M':
                _, ipd_batch,_ = self.gerdpipd(source_doa=source_doa)
            elif self.ch_mode == 'MM':
                _, ipd_batch,_ = self.gerdpipd(source_doa=source_doa)
            ipd_batch = np.concatenate((ipd_batch.real, ipd_batch.imag), axis=2).astype(np.float32) # (nb, ntime, 2nf, nmic-1, nsource)
            ipd_batch = torch.from_numpy(ipd_batch)

            vad_batch = vad_batch.mean(axis=2).float() # (nb,nseg,nsource) # s>2/3

            # DOAw_batch = torch.from_numpy(source_doa).to(self.device)
            DOAw_batch = DOAw_batch.to(self.device) # (nb,nseg,2,nsource)
            ipd_batch = ipd_batch.to(self.device)
            vad_batch = vad_batch.to(self.device)

            if self.tar_useVAD:
                nb, nt, nf, nmic, num_source = ipd_batch.shape
                th = 0
                vad_batch_copy = deepcopy(vad_batch)
                vad_batch_copy[vad_batch_copy<=th] = th
                vad_batch_copy[vad_batch_copy>0] = 1
                vad_batch_expand = vad_batch_copy[:, :, np.newaxis, np.newaxis, :].expand(nb, nt, nf, nmic, num_source)
                ipd_batch = ipd_batch * vad_batch_expand
            ipd_batch = torch.sum(ipd_batch, dim=-1)  # (nb,nseg,2nf,nmic-1)

            gt_batch['doa'] = DOAw_batch
            gt_batch['ipd'] = ipd_batch
            gt_batch['vad_sources'] = vad_batch

        return dp_data, data, dp_data_mag, data_mag, dp_data_phase, data_phase, dp_stft_rebatch_norm, stft_rebatch_norm, gt_batch, data_RIRI, dp_data_RIRI # [Input, DOA, IPD, VAD]

    def ce_loss(self, pred_batch=None, gt_batch=None):
        """
        Function: ce loss
        Args:
            pred_batch: doa
            gt_batch: dict{'doa'}
        Returns:
            loss
        """
        pred_doa = pred_batch
        gt_doa = gt_batch['doa'] * 180 / np.pi
        gt_doa = gt_doa[:,:,1,:].type(torch.LongTensor).to(self.device)
        nb,nt,_ = pred_doa.shape
        pred_doa = pred_doa.to(self.device)
        loss = torch.nn.functional.cross_entropy(pred_doa.reshape(nb*nt,-1),gt_doa.reshape(nb*nt))
        return loss
    def mse_loss(self, pred_batch = None, gt_batch = None):
        """
        Function: mse loss
        Args:
            pred_batch: ipd
            gt_batch: dict{'ipd'}
        Returns:
            loss
        """
        pred_ipd = pred_batch
        gt_ipd = gt_batch['ipd']
        nb, _, _, _ = gt_ipd.shape # (nb, nt, nf, nmic)

        pred_ipd_rebatch = self.removebatch(pred_ipd, nb).permute(0, 2, 3, 1)

        loss_doa = torch.nn.functional.mse_loss(pred_ipd_rebatch.contiguous(), gt_ipd.contiguous())

        return  loss_doa



    def predgt2DOA_cls(self, pred_batch=None, gt_batch=None):
        """
        Function: pred to doa of classification
        Args:
            pred_batch: doa classification
        Returns:
            loss
        """		
        if pred_batch is not None:
            pred_batch = pred_batch.detach()
            DOA_batch_pred = torch.argmax(pred_batch,dim=-1) # distance = 1 (nb, nt, 2)
            pred_batch = [DOA_batch_pred[:, :, np.newaxis, np.newaxis]]  # !! only for single source
        return pred_batch, gt_batch

    def predgt2DOA(self, pred_batch=None, gt_batch=None, time_pool_size=None):
        """
        Function: Conert IPD vector to DOA
        Args:
            pred_batch: ipd
            gt_batch: dict{'doa', 'vad_sources', 'ipd'}
        Returns:
            pred_batch: dict{'doa', 'spatial_spectrum'}
            gt_batch: dict{'doa', 'vad_sources', 'ipd'}
        """

        if pred_batch is not None:

            pred_ipd = pred_batch.detach()
            dpipd_template, _, doa_candidate = self.gerdpipd( ) # (nele, nazi, nf, nmic)

            _, _, _, nmic = dpipd_template.shape
            nbnmic, nt, nf = pred_ipd.shape
            nb = int(nbnmic/nmic)

            dpipd_template = np.concatenate((dpipd_template.real, dpipd_template.imag), axis=2).astype(np.float32) # (nele, nazi, 2nf, nmic-1)
            dpipd_template = torch.from_numpy(dpipd_template).to(self.device) # (nele, nazi, 2nf, nmic)

            # !!!
            nele, nazi, _, _ = dpipd_template.shape
            dpipd_template = dpipd_template[int((nele-1)/2):int((nele-1)/2)+1, int((nazi-1)/2):nazi, :, :]
            doa_candidate[0] = np.linspace(np.pi/2, np.pi/2, 1)
            doa_candidate[1] = np.linspace(0, np.pi, 37)
            # doa_candidate[0] = doa_candidate[0][int((nele-1)/2):int((nele-1)/2)+1]
            # doa_candidate[1] = doa_candidate[1][int((nazi-1)/2):nazi]

            # rebatch from (nb*nmic, nt, 2nf) to (nb, nt, 2nf, nmic)
            pred_ipd_rebatch = self.removebatch(pred_ipd, nb).permute(0, 2, 3, 1) # (nb, nt, 2nf, nmic)(1,24,514,1)
            if time_pool_size is not None:
                nt_pool = int(nt / time_pool_size)
                ipd_pool_rebatch = torch.zeros((nb, nt_pool, nf, nmic), dtype=torch.float32, requires_grad=False).to(self.device)  # (nb, nt_pool, 2nf, nmic-1)
                for t_idx in range(nt_pool):
                    ipd_pool_rebatch[:, t_idx, :, :]  = torch.mean(
                    pred_ipd_rebatch[:, t_idx*time_pool_size: (t_idx+1)*time_pool_size, :, :], dim=1)
                pred_ipd_rebatch = deepcopy(ipd_pool_rebatch)
                nt = deepcopy(nt_pool)

            pred_DOAs, pred_VADs, pred_ss = self.sourcelocalize(pred_ipd=pred_ipd_rebatch, dpipd_template=dpipd_template, doa_candidate=doa_candidate)
            pred_batch = {}
            pred_batch['doa'] = pred_DOAs
            pred_batch['vad_sources'] = pred_VADs
            pred_batch['spatial_spectrum'] = pred_ss

        if gt_batch is not None:
            for key in gt_batch.keys():
                gt_batch[key] = gt_batch[key].detach()

        return pred_batch, gt_batch

    def evaluate(self, pred, gt, metric_setting={'ae_mode':['azi'], 'ae_TH':5, 'useVAD':True, 'vad_TH':[2/3, 2/3], 'metric_unfold':False},idx=None ):
        """
        Function: Evaluate DOA estimation results
        Args:
            pred 	- dict{'doa', 'vad_sources'}
            gt 		- dict{'doa', 'vad_sources'}
                            doa (nb, nt, 2, nsources) in radians
                            vad (nb, nt, nsources) binary values
        Returns:
            metric
        """
        doa_gt = gt['doa'] * 180 / np.pi
        doa_pred = pred['doa'] * 180 / np.pi
        vad_gt = gt['vad_sources']
        vad_pred = pred['vad_sources']
        if idx != None:
            save_path = './locata_result/'
            np.save(save_path+str(idx)+'_gt',doa_gt.cpu().numpy())
            np.save(save_path+str(idx)+'_est',doa_pred.cpu().numpy())
            np.save(save_path+str(idx)+'_vadgt',vad_gt.cpu().numpy())
        # single source
        # metric = self.getmetric(doa_gt, vad_gt, doa_pred, vad_pred, ae_mode = ae_mode, ae_TH=ae_TH, useVAD=False, vad_TH=vad_TH, metric_unfold=Falsemetric_unfold)

        # multiple source
        metric = \
            self.getmetric(doa_gt, vad_gt, doa_pred, vad_pred,
                ae_mode = metric_setting['ae_mode'], ae_TH=metric_setting['ae_TH'],
                useVAD=metric_setting['useVAD'], vad_TH=metric_setting['vad_TH'],
                metric_unfold=metric_setting['metric_unfold'])
        nb, _, _, _ = doa_gt.shape
        angle_diff = self.getmetric.angular_error(doa_gt, doa_pred, 'azi')
        rmse = torch.sqrt(
            torch.nn.functional.mse_loss(angle_diff.view(nb, -1), torch.zeros_like(angle_diff.view(nb, -1))))
        metric['rmse'] = rmse
        return metric

    def data_preprocess_predict(self, mic_sig_batch=None, gt_batch=None, vad_batch=None, eps=1e-6, nor_flag=True):

        data = []
        dp_data = []
        if mic_sig_batch is not None:
            mic_sig_batch = mic_sig_batch.to(self.device)

            stft = self.dostft(signal=mic_sig_batch)  # (nb,nf,nt,nch)
            stft = stft.permute(0, 3, 1, 2)  # (nb,nch,nf,nt)

            # change batch (nb,nch,nf,nt)→(nb*(nch-1),2,nf,nt)/(nb*(nch-1)*nch/2,2,nf,nt)
            stft_rebatch = self.addbatch(stft)
            if nor_flag:
                nb, nc, nf, nt = stft_rebatch.shape
                mag = torch.abs(stft_rebatch)
                mean_value = forgetting_norm(mag)
                stft_rebatch_real = torch.real(stft_rebatch) / (mean_value + eps)
                stft_rebatch_image = torch.imag(stft_rebatch) / (mean_value + eps)
            else:
                stft_rebatch_real = torch.real(stft_rebatch)
                stft_rebatch_image = torch.imag(stft_rebatch)

            # prepare model input
            real_image_batch = torch.cat((stft_rebatch_real, stft_rebatch_image), dim=1)
            data += [real_image_batch[:, :, self.fre_range_used, :]]

        if gt_batch is not None:
            DOAw_batch = gt_batch['doa']
            vad_batch = gt_batch['vad_sources']

            source_doa = DOAw_batch.cpu().numpy()

            if self.ch_mode == 'M':
                _, ipd_batch, _ = self.gerdpipd(source_doa=source_doa)
            elif self.ch_mode == 'MM':
                _, ipd_batch, _ = self.gerdpipd(source_doa=source_doa)
            ipd_batch = np.concatenate(
                (ipd_batch.real[:, :, self.fre_range_used, :, :], ipd_batch.imag[:, :, self.fre_range_used, :, :]),
                axis=2).astype(np.float32)  # (nb, ntime, 2nf, nmic-1, nsource)
            ipd_batch = torch.from_numpy(ipd_batch)

            vad_batch = vad_batch.mean(axis=2).float()  # (nb,nseg,nsource) # s>2/3

            # DOAw_batch = torch.from_numpy(source_doa).to(self.device)
            DOAw_batch = DOAw_batch.to(self.device)  # (nb,nseg,2,nsource)
            ipd_batch = ipd_batch.to(self.device)
            vad_batch = vad_batch.to(self.device)

            if self.tar_useVAD:
                nb, nt, nf, nmic, num_source = ipd_batch.shape
                th = 0
                vad_batch_copy = deepcopy(vad_batch)
                vad_batch_copy[vad_batch_copy <= th] = th
                vad_batch_copy[vad_batch_copy > 0] = 1
                vad_batch_expand = vad_batch_copy[:, :, np.newaxis, np.newaxis, :].expand(nb, nt, nf, nmic, num_source)
                ipd_batch = ipd_batch * vad_batch_expand
            ipd_batch = torch.sum(ipd_batch, dim=-1)  # (nb,nseg,2nf,nmic-1)

            gt_batch['doa'] = DOAw_batch
            gt_batch['ipd'] = ipd_batch
            gt_batch['vad_sources'] = vad_batch

            data += [gt_batch]

        return data  # [Input, DOA, IPD, VAD]
