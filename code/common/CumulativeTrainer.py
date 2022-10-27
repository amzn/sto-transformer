"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
from torch.utils.data.distributed import DistributedSampler
from common.Utils import *
from common.EMA import *
from tqdm import tqdm


def train_embedding(model):
    for name, param in model.named_parameters():
        if 'embedding' in name:
            print('requires_grad', name, param.size())
            param.requires_grad = True


def init_params(model, escape=None):
    for name, param in model.named_parameters():
        if escape is not None and escape in name:
            print('no_init', name, param.size())
            continue
        print('init', name, param.size())
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        # if 'bias' in name:
        #     constant_(param.data, 0)
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class CumulativeTrainer(object):
    def __init__(self, model, tokenizer, detokenizer, local_rank, num_gpus, accumulation_steps=1, ema_rate=0.995, max_grad_norm=1, save_data_attributes=None):
        super(CumulativeTrainer, self).__init__()
        self.local_rank=local_rank
        self.num_gpus=num_gpus
        self.tokenizer=tokenizer
        self.detokenizer=detokenizer

        if local_rank is not None:
            torch.cuda.set_device(local_rank)

        if torch.cuda.is_available():
            self.model =model.cuda()
        else:
            self.model = model

        self.accumulation_steps=accumulation_steps
        self.accumulation_count=0

        if torch.cuda.is_available() and local_rank is not None:
            print("GPU ", self.local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        self.ema = EMA(self.model, ema_rate)
        self.ema.register()

        self.max_grad_norm = max_grad_norm
        self.save_data_attributes = save_data_attributes # save parts of data

    def train_batch(self, epoch, data, method, optimizer, scheduler=None):
        self.accumulation_count+=1
        loss = self.model(data, method=method)

        if isinstance(loss, tuple) or isinstance(loss, list):
            closs = [l.mean().cpu().item() for l in loss]
            loss = torch.cat([l.mean().reshape(1) for l in loss]).sum()
            # loss = torch.cat(loss, dim=-1).mean()
        else:
            loss = loss.mean()
            closs = [loss.cpu().item()]

        loss = loss/self.accumulation_steps
        loss.backward()

        if self.accumulation_count % self.accumulation_steps == 0:
            # for name, param in self.model.named_parameters():
            #     print(param.grad)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            optimizer.step()
            self.ema.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        return closs

    def serialize(self, epoch, output_path, use_multiple_gpu=torch.cuda.is_available()):
        if self.local_rank!=0 and use_multiple_gpu:
            return
        # output_path = os.path.join(output_path, 'model/')
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        this_model = self.model.module if use_multiple_gpu else self.model
        torch.save(this_model.state_dict(), os.path.join(output_path, '.'.join([str(epoch), 'pkl'])))

    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch, optimizer, scheduler=None):
        self.model.train()

        if torch.cuda.is_available() and self.num_gpus > 1:
            sampler = DistributedSampler(train_dataset)
            sampler.set_epoch(epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, sampler=sampler, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True)

        start_time = time.time()
        count_batch=0
        ep_loss_list = []
        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda
            count_batch += 1

            bloss = self.train_batch(epoch, data, method=method, optimizer=optimizer, scheduler=scheduler)
            if self.local_rank is None or self.local_rank in [-1, 0]:
                if j >= 0 and j % self.accumulation_steps == 0:
                    elapsed_time = time.time() - start_time
                    if scheduler is not None:
                        print('Method', method, 'Epoch', epoch, 'Batch ', count_batch, 'Loss ', bloss, 'Time ', elapsed_time, 'Learning rate ', scheduler.get_last_lr())
                    else:
                        print('Method', method, 'Epoch', epoch, 'Batch ', count_batch, 'Loss ', bloss, 'Time ', elapsed_time)
                    ep_loss_list.extend(bloss)
                sys.stdout.flush()

        if self.accumulation_count % self.accumulation_steps != 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        # elapsed_time = time.time() - start_time
        # if self.local_rank is None or self.local_rank == 0:
        #     if scheduler is not None:
        #         print('Method', method, 'Epoch', epoch, 'Batch ', count_batch, 'Loss ', bloss, 'Time ', elapsed_time, 'Learning rate ', scheduler.get_last_lr())
        #     else:
        #         print('Method', method, 'Epoch', epoch, 'Batch ', count_batch, 'Loss ', bloss, 'Time ', elapsed_time)
        # sys.stdout.flush()
        return ep_loss_list

    def predict(self, method, dataset, collate_fn, batch_size, epoch):
        self.model.eval()
        rs = []
        losses = []
        with torch.no_grad():
            if torch.cuda.is_available() and self.num_gpus > 1:
                sampler = DistributedSampler(dataset, shuffle=False)
                test_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, sampler=sampler, pin_memory=True)
            else:
                test_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True)

            print('Method=%s' % method, 'Epoch=%s' % epoch, 'Batch=%s'%batch_size, 'DataLoaderLen=%s' % len(test_loader))
            for k, data_cpu in tqdm(enumerate(test_loader, 0), disable=True):
                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data_cpu.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda
                else:
                    data = data_cpu
                # golden_data = data # can not store all due to OOM
                if self.save_data_attributes:
                    golden_data = {}
                    for k in self.save_data_attributes:
                        golden_data[k] = data[k]
                else:
                    golden_data = data
                    # golden_data['neighbor_context'] = data['neighbor_context'][:2]

                pred_output = self.model(data, method=method)
                rs.append([golden_data, pred_output])
                loss = self.model(data, method='train')
                if isinstance(loss, tuple) or isinstance(loss, list):
                    closs = [l.mean().cpu().item() for l in loss]
                else:
                    closs = [loss.cpu().item()]
                losses.append(closs)
        return rs, losses


