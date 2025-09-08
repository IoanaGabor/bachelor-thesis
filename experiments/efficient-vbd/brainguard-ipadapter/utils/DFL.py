# This file is an adaptation of https://github.com/kunzhan/BrainGuard/blob/main/utils/DFL.py

import wandb
import numpy as np
import torch
import torch.nn as nn
import copy
from utils.loss_avg import AverageMeter


class DFL:
    def __init__(self,
                cid: int,
                cuda_id: int, 
                soft_clip_loss: nn.Module,
                logit_scale: float,
                train_dl,
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 50,
                num_pre_loss: int = 10,
                ) -> None:

        self.cid = cid
        self.soft_clip_loss = soft_clip_loss
        self.logit_scale = logit_scale
        self.train_dl = train_dl
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device
        self.cuda_id = cuda_id
        self.weights = None 
        self.start_phase = True
        self.losses_value = AverageMeter()
        self.all_steps = 0
        self.loss_mse = nn.MSELoss(reduction='mean').to(f'cuda:{self.cuda_id}')

    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module,
                            writer,
                            round,
                            ) -> None:
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        for param, param_g in zip(params[2:-self.layer_idx], params_g[2:-self.layer_idx]):
            param.data.copy_(param_g.data)

        model_t = copy.deepcopy(local_model)
        model_t.to(f'cuda:{self.cuda_id}')
        params_t = list(model_t.parameters())

        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        optimizer = torch.optim.AdamW(params_tp, lr=0)

        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(f'cuda:{self.cuda_id}') for param in params_p]

        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_gpu = param.to(f'cuda:{self.cuda_id}')
            param_g_gpu = param_g.to(f'cuda:{self.cuda_id}')
            param_t.data = param_gpu + (param_g_gpu - param_gpu) * weight

        losses = []  
        cnt = 0  
        while True: 
            for train_i, data_i in enumerate(self.train_dl):
                repeat_index = train_i % 3 
                voxel, _, ip_emb = data_i
                voxel = torch.mean(voxel,axis=1)
                voxel = voxel.to(f'cuda:{self.cuda_id}').float()
                clip_image = ip_emb
                clip_image = clip_image.to(f'cuda:{self.cuda_id}')
                ridge_out = model_t.ridge(voxel)
                results = model_t.backbone(ridge_out)

                optimizer.zero_grad()

                clip_image_pred = results
                clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
                clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)
                loss_mse_image = self.loss_mse(clip_image_pred_norm, clip_image_norm) * 10000
                loss_clip_image = self.soft_clip_loss(
                    clip_image_pred_norm,
                    clip_image_norm,
                    )
                
                loss =  loss_mse_image * 2 + loss_clip_image
                self.losses_value.update(loss.item())
                if (train_i % (len(self.train_dl) // 8) == 0):
                    print(f"client{self.cid}: loss_DFL: {self.losses_value.avg:.4f}")
                    #wandb.log({'client{self.cid}loss_DFL':self.losses_value.avg}, step=self.all_steps*len(self.train_dl)+train_i)
                loss.backward()
                writer.add_scalar(f'Loss_DFL/client_{self.cid}', self.losses_value.avg, self.all_steps * len(self.train_dl) + train_i)

                for param_t, param, param_g, weight in zip(params_tp, params_p,params_gp, self.weights):

                    param_gpu = param.to(f'cuda:{self.cuda_id}')
                    param_g_gpu = param_g.to(f'cuda:{self.cuda_id}')
                    weight_update = self.eta * ((param_t.grad * 1000) * (param_g_gpu - param_gpu))
                    weight.data = torch.clamp(weight.data - weight_update, 0, 1)

                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_gpu = param.to(f'cuda:{self.cuda_id}')
                    param_g_gpu = param_g.to(f'cuda:{self.cuda_id}')
                    param_t.data = param_gpu + (param_g_gpu - param_gpu) * weight
            self.all_steps += 1        
            losses.append(loss.item())
            cnt += 1
            
            if not self.start_phase:
                break

            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                    '\tDFL epochs:', cnt)
                break

        self.start_phase = False

        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone().to('cpu')