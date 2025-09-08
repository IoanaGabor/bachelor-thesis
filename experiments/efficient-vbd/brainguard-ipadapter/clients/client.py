
# This file is an adaptation of https://github.com/kunzhan/BrainGuard/blob/main/clients/client.py

import wandb
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from utils.DFL import DFL
from utils.loss_avg import AverageMeter
from utils.utils import soft_clip_loss
import math
from trainmodel.models import *
import utils.data as data
from utils.utils import prepare_coco

class Client(object):
    def __init__(self, args, id, train_samples, cuda_id):

        self.cuda_id = cuda_id
        self.model = copy.deepcopy(args.model)
        self.model.ridge = RidgeRegression(input_size=args.multi_voxel_dims[id], out_features=2048)
        self.model = self.model.to('cuda:{}'.format(self.cuda_id))
        self.model_ema = copy.deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        self.dataset = args.dataset
        self.device = 'cuda:{}'.format(self.cuda_id)
        self.id = id
        self.args = args
        self.flag_dfl=True
        self.train_samples = train_samples
        self.batch_size = args.batch_size
        self.global_rounds = args.global_rounds
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.set_opt_grouped_parameters(args)
        self.loss_mse = nn.MSELoss(reduction='mean').to(f'cuda:{self.cuda_id}')
        self.optimizer = torch.optim.AdamW(self.opt_grouped_parameters, betas=(0.9, 0.9999), lr=self.learning_rate, eps=1e-8)
        self.train_type = args.train_type
        self.prompts_list = prepare_coco(args.data_root)

        self.prepare_dataloader()
        self.set_lr_scheduler(args)

        self.eta = args.eta
        self.layer_idx = args.layer_idx
        print("client train_dl")
        print(self.train_dl)
        self.DFL = DFL(self.id, self.cuda_id, soft_clip_loss, 0.1, self.train_dl, self.layer_idx, self.eta, self.device)
        
        self.global_best_val_sim_image= 0.
        self.global_model_best_val_sim_image= 0.
        self.all_steps = 0
        self.best_val_bwd = 0.
        self.flag_ala = True
        self.before_aggregate_bwd = 0.

        self.total_loss = AverageMeter()
        self.mse_image = AverageMeter()
        self.nce_image = AverageMeter()
        
    def train(self, writer, round, logger):
        self.model.to(f'cuda:{self.cuda_id}')
        self.model_ema.to(f'cuda:{self.cuda_id}')
        self.model.train()

        for step in range(self.local_steps):

            logger.info("Start train Client {}, global_round: {}/{}  Local step:{}/{}".format(self.id, round+1, self.global_rounds, step+1, self.local_steps))

            for train_i, data_i in enumerate(self.train_dl):
                self.train_i = train_i
                repeat_index = train_i % 3
                voxel, _, ip= data_i
                voxel = voxel[:,repeat_index,...].float()

                clip_image = ip

                voxel = voxel.to(f'cuda:{self.cuda_id}')
                clip_image = ip.to(f'cuda:{self.cuda_id}')
                
                ridge_out = self.model.ridge(voxel)
                results = self.model.backbone(ridge_out)

                clip_image_pred = results
                clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
                clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)

                loss_mse_image = self.loss_mse(clip_image_pred_norm, clip_image_norm) * 10000

                loss_clip_image = soft_clip_loss(
                    clip_image_pred_norm,
                    clip_image_norm,
                )
                
                loss =  loss_mse_image * 2 + loss_clip_image
                self.update_local_ema(self.model, self.model_ema, self.args.ema_decay)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                current_lr = self.lr_scheduler.get_last_lr()[0]
                self.total_loss.update(loss.item())
                self.mse_image.update(loss_mse_image.item())
                self.nce_image.update(loss_clip_image.item())

                writer.add_scalar(f'Loss/loss_All_train_client_{self.id}', self.total_loss.avg, self.all_steps * len(self.train_dl) + train_i)
                writer.add_scalar(f'Loss/loss_Mse_image_train_client_{self.id}', self.mse_image.avg, self.all_steps * len(self.train_dl) + train_i)
                writer.add_scalar(f'Loss/loss_SoftCliptrain_image_client_{self.id}', self.nce_image.avg, self.all_steps * len(self.train_dl) + train_i)
                writer.add_scalar(f'Learning rate/train_client_{self.id}', current_lr, self.all_steps * len(self.train_dl) + train_i)

                if (train_i % (len(self.train_dl) // 8) == 0):
                    logger.info(f"client{self.id}: Learning rate: {current_lr:.4f}, Loss softclip image:{self.nce_image.avg:.4f}")
                

                if (train_i % (len(self.train_dl) // 8) == 0):
                    logger.info(f"client{self.cuda_id}: Learning rate: {current_lr:.4f}, Loss softclip image:{self.nce_image.avg:.4f}")

            self.all_steps += 1
            self.model.eval()

            with torch.no_grad():
                val_sims_base_image = AverageMeter()
                val_loss_base_mse_image = AverageMeter()
                val_loss_base_nce_image = AverageMeter()

                for val_i, data_i in enumerate(self.val_dl):
                    self.val_i = val_i
                    repeat_index = val_i % 3
                    voxel, image, coco, ip = data_i
                    voxel = torch.mean(voxel,axis=1)
                    voxel = voxel.to(f'cuda:{self.cuda_id}').float()
                    clip_image = ip

                    clip_image = clip_image.to(f'cuda:{self.cuda_id}')
                    ridge_out = self.model.ridge(voxel)
                    results = self.model.backbone(ridge_out)

                    clip_image_pred = results
                    clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
                    clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)

                    val_loss_mse_image = self.loss_mse(clip_image_pred_norm, clip_image_norm) * 10000

                    loss_clip_image = soft_clip_loss(
                        clip_image_pred_norm,
                        clip_image_norm,
                    )
                    val_sims_image = nn.functional.cosine_similarity(clip_image_norm, clip_image_pred_norm).mean().item()
                    val_loss_base_nce_image.update(loss_clip_image.item())
                    val_loss_base_mse_image.update(val_loss_mse_image.item())
                    val_sims_base_image.update(val_sims_image)
                writer.add_scalar(f'Val/sim_image_{self.id}', val_sims_base_image.avg, self.all_steps)
                writer.add_scalar(f'Val/loss_mse_image{self.id}', val_loss_base_mse_image.avg, self.all_steps)
                writer.add_scalar(f'Val/loss_SoftClip_image{self.id}', val_loss_base_nce_image.avg, self.all_steps)
                logger.info(f'client{self.id}  Mean sim image: {val_sims_base_image.avg}')
                if val_sims_base_image.avg > self.global_best_val_sim_image:
                    self.global_best_val_sim_image = val_sims_base_image.avg
                    torch.save(self.model.state_dict(), './logs/model/client{}_best_{}_{}_.pth'.format(self.id, round, self.global_best_val_sim_image))
        logger.info("Train Client {} done".format(self.id))
    def eval_local_model(self,writer, i, logger, type):
        if type == 'val':
            dl = self.val_dl
        elif type == 'train':
            dl = self.train_dl
        self.model.to(f'cuda:{self.cuda_id}')
        self.model_ema.to(f'cuda:{self.cuda_id}')
        self.model.eval()
        with torch.no_grad():
            local_total_loss = AverageMeter()
            local_sims_base_image = AverageMeter()
            local_loss_base_mse_image = AverageMeter()
            local_loss_base_nce_image = AverageMeter()
            
            for val_i, data_i in enumerate(dl):
                self.val_i = val_i
                voxel, _ ,ip = data_i
                voxel = torch.mean(voxel,axis=1)
                voxel = voxel.to(f'cuda:{self.cuda_id}').float()
                clip_image = ip

                clip_image = clip_image.to(f'cuda:{self.cuda_id}')
                ridge_out = self.model.ridge(voxel)
                results = self.model.backbone(ridge_out)

                clip_image_pred = results
                clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
                clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)

                val_loss_mse_image = self.loss_mse(clip_image_pred_norm, clip_image_norm) * 10000

                loss_clip_image = soft_clip_loss(
                    clip_image_pred_norm,
                    clip_image_norm,
                )
                loss =  val_loss_mse_image * 2 + loss_clip_image
                val_sims_image = nn.functional.cosine_similarity(clip_image_norm, clip_image_pred_norm).mean().item()
                local_loss_base_nce_image.update(loss_clip_image.item())
                local_loss_base_mse_image.update(val_loss_mse_image.item())
                local_sims_base_image.update(val_sims_image)
                local_total_loss.update(loss.item())
            writer.add_scalar(f'{type}/total_loss_{self.id}', local_total_loss.avg, self.all_steps)
            writer.add_scalar(f'{type}/sim_image_{self.id}', local_sims_base_image.avg, self.all_steps)
            writer.add_scalar(f'{type}/loss_mse_image{self.id}', local_loss_base_mse_image.avg, self.all_steps)
            writer.add_scalar(f'{type}/loss_SoftClip_image{self.id}', local_loss_base_nce_image.avg, self.all_steps)
            logger.info(f'client{self.id}  Mean sim image: {local_sims_base_image.avg}')
        return local_total_loss.avg, local_loss_base_mse_image.avg, local_loss_base_nce_image.avg, local_sims_base_image.avg


    def local_initialization(self, received_global_model, writer, round):
        self.model.to(f'cpu')
        temp_global_model = copy.deepcopy(received_global_model)
        temp_global_model.to(f'cpu')
        if self.flag_dfl:
            self.DFL.adaptive_local_aggregation(temp_global_model, self.model, writer, round)


    def load_train_data(self, batch_size=None, is_train=True):

        if batch_size == None:
            batch_size = self.batch_size
        train_data, test_data = read_client_data(self.id, self.args.train_type)
        if is_train:
            return DataLoader(train_data, batch_size, drop_last=False, shuffle=True, num_workers=4)
        else:
            if self.resume:
                return DataLoader(test_data, batch_size=1, drop_last=False, shuffle=False, num_workers=4)
            else:
                return DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=4)
    
    def set_opt_grouped_parameters(self, args):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.opt_grouped_parameters = [
        {'params': [p for n, p in self.model.ridge.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in self.model.ridge.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in self.model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in self.model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},]


    def set_lr_scheduler(self, args):
        total_steps=((args.global_rounds * self.local_steps) * math.ceil(8859 / args.batch_size))
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.learning_rate,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/(args.global_rounds * self.local_steps)
        )

    def prepare_dataloader(self):
        self.train_dl, self.val_dl = data.get_dls(
            subject=self.id,
            data_path=self.args.data_root,
            batch_size=self.args.batch_size,
            val_batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pool_type='max',
            pool_num=8192,
            length=8859,
            seed=42,
        )
        self.num_batches = len(self.train_dl)

    def update_local_ema(self, local_model, ema_model, alpha):
        for param, ema_param in zip(local_model.parameters(), ema_model.parameters()):
            ema_param.data = alpha * param.data + (1 - alpha) * ema_param.data
