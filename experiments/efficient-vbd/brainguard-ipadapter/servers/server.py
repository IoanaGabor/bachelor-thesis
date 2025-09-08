
# This file is an adaptation of https://github.com/kunzhan/BrainGuard/blob/main/servers/server.py

import wandb
import copy
import numpy as np
import torch
import time
from clients.client import *
from threading import Thread
from torch.utils.tensorboard import SummaryWriter
import os


class Server(object):
    def __init__(self, args):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = [1,2,5]
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(len(self.num_clients) * self.join_ratio)
        self.cuda_id = args.cuda_id
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_loss = []
        self.rs_train_loss = []
        self.eval_gap = args.eval_gap
        self.writer = None
        self.set_clients(args, Client)
        if not args.resume:
            log_dir = './logs/{}/{}'.format(args.train_type, time.strftime("%b%d_%d-%H-%M", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)
            os.makedirs(log_dir, exist_ok=True)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {len(self.num_clients)}")
        print("Finished creating server and clients.")
        self.selected_clients = self.select_clients()
        self.Budget = []

    def safe_train(self, client, writer, round_idx, logger):
        try:
            client.train(writer, round_idx, logger)
        except Exception as e:
            logger.error(f"Exception in client.train: {e}", exc_info=True)

    def safe_eval_global_model(client, global_model, writer, round_idx, logger):
        try:
            client.eval_global_model(global_model, writer, round_idx, logger)
        except Exception as e:
            logger.error(f"Exception in client.train: {e}", exc_info=True)
 


    def train(self, args):
        for i in range(self.global_rounds):
            args.logger.info(f"============= Round: {i+1}th =============")
            args.logger.info(f"selected clients{self.selected_clients}")
            s_t = time.time()
            if i != 0 and i < self.global_rounds:
                self.send_models(i)
                for client in self.selected_clients:
                   for type in ['train', 'val']:
                    local_total_loss, local_mse_loss, local_nce_loss, local_sims = client.eval_local_model(self.writer, i, args.logger, type)
                    wandb.log({
                        f"{client.id}/{type}_total_loss": local_total_loss,
                        f"{client.id}/{type}_mse_loss": local_mse_loss,
                        f"{client.id}/{type}_nce_loss": local_nce_loss,
                        f"{client.id}/{type}_sims": local_sims,
                        "step": i
                    })
            thread_train = [Thread(target=self.safe_train, args=(client, self.writer, i, args.logger))
                    for client in self.selected_clients]
            args.logger.info(f"len train{len(thread_train)}")
            for t in thread_train:
                t.start()

            for t in thread_train:
                t.join()
                args.logger.info("joined")
            args.logger.info(f"global rounds {self.global_rounds}")
            if i < self.global_rounds:
                args.logger.info("started if")
                self.receive_models()
                self.aggregate_parameters()

    def set_clients(self, args, clientObj):
        for client in self.num_clients:
            client = clientObj(args, 
                            id=client, 
                            train_samples=8859, 
                            cuda_id=self.cuda_id[str(client)])
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self, round):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.local_initialization(self.global_model, self.writer, round)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model_ema)

    def add_parameters(self, w, client_model):
        client_model = client_model.to('cpu' if self.cuda_id["server"]==-1 else f'cuda:{self.cuda_id["server"]}')
        for (server_param_name, server_param), (_, client_param) in zip(self.global_model.named_parameters(), client_model.named_parameters()):
            if 'ridge' not in server_param_name:
                server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        self.global_model.to('cpu' if self.cuda_id["server"]==-1 else f'cuda:{self.cuda_id["server"]}')
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
