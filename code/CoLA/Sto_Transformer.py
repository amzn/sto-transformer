"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import batch_to_ids,Elmo
import numpy as np
import os.path
import pdb

class Embeddings(nn.Module):
        def __init__(self,vocab_size,max_len,
                     h_size,h_attn_size,
                     use_elmo,num_rep,elmo_drop, gpu_id=-1):
                super(Embeddings,self).__init__()
                self.use_elmo=use_elmo
                
                self.token_embeds=nn.Embedding(vocab_size,h_size,padding_idx=0)
                self.pos_embeds=nn.Embedding(max_len,h_size+use_elmo*1024)
                self.layer_norm=nn.LayerNorm(use_elmo*1024+h_size)
                
                options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" 
                weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
                if use_elmo:
                        self.elmo=Elmo(options_file,weight_file,num_rep,dropout=elmo_drop)

                self.project=nn.Linear(use_elmo*1024+h_size,h_attn_size)
                self.gpu_id = gpu_id
                
        def forward(self,input,pos,data=None):
                if self.use_elmo:
                        character_ids=batch_to_ids(data).cuda(self.gpu_id) if self.gpu_id > -1 else batch_to_ids(data)
                        rep=self.elmo(character_ids)['elmo_representations'][0]
                        input=self.token_embeds(input)
                        rep=torch.cat([rep,input],dim=-1)
                else:
                        rep=self.token_embeds(input)
                pos=self.pos_embeds(pos)
                
                output=self.layer_norm(rep+pos)
                output=self.project(output)
                
                return output


class StoSelfAttention(nn.Module):
        def __init__(self, h_size, n_heads, prob_attn, prob_h, tau2):
                super(StoSelfAttention, self).__init__()
                self.n_heads = n_heads
                self.h_size = h_size
                self.head_dim = self.h_size // self.n_heads

                self.query = nn.Linear(h_size, h_size)
                self.key = nn.Linear(h_size, h_size)
                self.value = nn.Linear(h_size, h_size)

                self.dropout_attn = nn.Dropout(p=prob_attn)
                self.dropout_h = nn.Dropout(p=prob_h)
                self.out = nn.Linear(h_size, h_size)

                self.layer_norm = nn.LayerNorm(h_size)

                self.tau2 = tau2


        def forward(self, input, input_mask):
                qq = self.query(input)
                kk = self.key(input)
                vv = self.value(input)

                qq = qq.view(input.shape[0], -1, self.n_heads, self.head_dim)
                kk = kk.view(input.shape[0], -1, self.n_heads, self.head_dim)

                vv = vv.view(input.shape[0], -1, self.n_heads, self.head_dim)

                qq = qq.transpose(1, 2)
                kk = kk.transpose(1, 2)
                vv = vv.transpose(1, 2)

                interact = torch.matmul(qq, kk.transpose(-1, -2))
                # attn_weights=F.softmax(interact,dim=-1)
                sto_attn_weights = F.gumbel_softmax(interact, tau=self.tau2, hard=False, dim=3)
                mask_1 = input_mask.unsqueeze(-1).unsqueeze(1).expand(-1, self.n_heads, -1, input.shape[1])
                mask_2 = input_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.n_heads, input.shape[1], -1)
                attn_weights = sto_attn_weights * (mask_1 * mask_2)
                attn_weights = self.dropout_attn(attn_weights)

                output = torch.matmul(attn_weights, vv)
                output = output.transpose(1, 2)
                output = output.contiguous().view(input.shape[0], -1, self.h_size)

                output = self.dropout_h(self.out(output))
                output = self.layer_norm(output + input)

                return output

class DualStoSelfAttention(nn.Module):
        def __init__(self,h_size,n_heads,prob_attn,prob_h, tau1, tau2, n_centroids):
                super(DualStoSelfAttention,self).__init__()
                self.n_heads=n_heads
                self.h_size=h_size
                self.head_dim = self.h_size // self.n_heads
                
                self.query=nn.Linear(h_size,h_size)
                self.key=nn.Linear(h_size,h_size)
                self.value=nn.Linear(h_size,h_size)

                self.dropout_attn=nn.Dropout(p=prob_attn)
                self.dropout_h=nn.Dropout(p=prob_h)
                self.out=nn.Linear(h_size,h_size)
                
                self.layer_norm=nn.LayerNorm(h_size)

                self.tau1=tau1
                self.tau2=tau2
                self.n_centroids = n_centroids

                self.centroid = torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(self.head_dim, self.n_centroids), a=-0.5, b=0.5),
                                           requires_grad=True)

                #self.q_centroid = torch.nn.Parameter(
                #        torch.nn.init.uniform_(torch.empty(self.head_dim, self.n_centroids), a=-0.5, b=0.5),
                #        requires_grad=True)
                
        def forward(self,input,input_mask):
                qq=self.query(input)
                kk=self.key(input)
                vv=self.value(input)


                qq=qq.view(input.shape[0],-1,self.n_heads,self.head_dim)
                kk=kk.view(input.shape[0],-1,self.n_heads,self.head_dim)

                '''
                
                qq_ = torch.einsum("nshd,dc->nshc", [qq, self.q_centroid])
                q_prob = F.gumbel_softmax(qq_, tau=self.tau1, hard=False, dim=-1)
                #pdb.set_trace()
                sto_qq = torch.einsum("nshc,cd->nshd", [q_prob, self.q_centroid.T])
                sto_qq = sto_qq.view(input.shape[0],-1,self.n_heads,self.head_dim)
                '''

                #pdb.set_trace()
                #kk = kk.view(input.shape[0],-1,self.n_heads * self.head_dim)
                kk_ = torch.einsum("nshd,dc->nshc", [kk, self.centroid])
                prob = F.gumbel_softmax(kk_, tau=self.tau1, hard=True, dim=-1)
                #pdb.set_trace()
                sto_kk = torch.einsum("nshc,cd->nshd", [prob, self.centroid.T])
                sto_kk = sto_kk.view(input.shape[0],-1,self.n_heads,self.head_dim)
                #pdb.set_trace()

                vv=vv.view(input.shape[0],-1,self.n_heads,self.head_dim)
                
                qq=qq.transpose(1,2)
                sto_kk=sto_kk.transpose(1,2)
                vv=vv.transpose(1,2)

                #interact=torch.matmul(qq, sto_kk.transpose(-1,-2))
                interact = torch.matmul(qq, sto_kk.transpose(-1, -2))
                #attn_weights=F.softmax(interact,dim=-1)
                #for i in range(0, 10):
                sto_attn_weights = F.gumbel_softmax(interact, tau=self.tau2, hard=True, dim=3)
                mask_1=input_mask.unsqueeze(-1).unsqueeze(1).expand(-1,self.n_heads,-1,input.shape[1])
                mask_2=input_mask.unsqueeze(1).unsqueeze(1).expand(-1,self.n_heads,input.shape[1],-1)
                attn_weights=sto_attn_weights * (mask_1*mask_2)
                attn_weights=self.dropout_attn(attn_weights)

                output=torch.matmul(attn_weights,vv)
                output=output.transpose(1,2)
                output=output.contiguous().view(input.shape[0],-1,self.h_size)
                
                output=self.dropout_h(self.out(output))
                output=self.layer_norm(output+input)
                #print(attn_weights.detach().cpu().numpy()[0][0])
                #pdb.set_trace()
                return output
            
            
class Intermediate(nn.Module):
        def __init__(self,inter_size,h_size):
                super(Intermediate,self).__init__()
                
                self.linear=nn.Linear(h_size,inter_size)
                self.act=nn.GELU()
                
        def forward(self,input):
                output=self.linear(input)
                output=self.act(output)
                
                return output
            
            
class FFN(nn.Module):
        def __init__(self,h_size,inter_size):
                super(FFN,self).__init__()
                
                self.linear=nn.Linear(inter_size,h_size)
                self.layernorm=nn.LayerNorm(h_size)
                
        def forward(self,input,attn_output):
                output=self.linear(input)
                output=self.layernorm(output+attn_output)
                
                return output
            
            
class Layer(nn.Module):
        def __init__(self,h_size,inter_size,
                     n_heads,prob_attn,prob_h, tau1, tau2, centroids, dual):
                super(Layer,self).__init__()
                if dual:
                        self.attn =DualStoSelfAttention(h_size,n_heads,prob_attn,prob_h, tau1, tau2, centroids)
                else:
                        self.attn =StoSelfAttention(h_size, n_heads, prob_attn, prob_h, tau1)
                self.inter=Intermediate(inter_size,h_size)
                self.ffn=FFN(h_size,inter_size)
                
        def forward(self,input,input_mask):
                attn=self.attn(input,input_mask)
                inter=self.inter(attn)
                output=self.ffn(inter,attn)
                
                return output
            

class Pooler(nn.Module):
        def __init__(self,h_size,prob,n_options=2):
                super(Pooler,self).__init__()
                
                self.project=nn.Linear(h_size,n_options)
                self.dropout=nn.Dropout(p=prob)
                
        def forward(self,input):
                output=input[:,0,:].view(input.shape[0],1,-1)
                output=self.dropout(output)
                output=self.project(output).squeeze(1)
                
                return output

           
class Model_S(nn.Module):
        def __init__(self, dual, embed_size,h_size,inter_size,vocab_size,
                     max_len,n_heads,n_layers,per_layer,prob_cl,prob_attn,prob_h,
                     use_elmo, tau1, tau2, centroids, num_rep=None,elmo_drop=None, gpu_id=-1):
                super(Model_S,self).__init__()
                self.embed=Embeddings(vocab_size,max_len,
                                      embed_size,h_size,
                                      use_elmo,num_rep,elmo_drop, gpu_id)
                #if dual:
                self.layer=nn.ModuleList([Layer(h_size,inter_size,
                                                n_heads,prob_attn,prob_h,
                                                tau1, tau2, centroids, dual) for _ in range(n_layers)])

                self.per_layer=per_layer
                
                self.pooler=Pooler(h_size,prob_cl,2)
                
        def forward(self,token,pos,input_mask,data=None):
                output=self.embed(token,pos,data)

                for layer in self.layer:
                        for _ in range(self.per_layer):
                                output=layer(output,input_mask)
         
                output=self.pooler(output)
                                
                return output