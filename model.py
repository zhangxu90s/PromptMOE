# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
import typing as tp
import torch.nn.functional as F
import numpy as np
import copy

import random


    
class Model(nn.Module):   
    def __init__(self, encoder,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.viewer_number = 2
        
        self.num = 4
        self.dropout_ops = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(self.num)
        ])
        
        self.dnn = nn.Sequential(nn.Linear(768, self.num),
                         nn.Softmax())

    def forward(self, code_inputs=None, nl_inputs=None, training=False): 

        if code_inputs is not None and nl_inputs is None:
            '''
            if training:
                output = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]

                logits = []
                for i, dropout_op in enumerate(self.dropout_ops):
                    temp_out = dropout_op(output)
                    logits.append(temp_out) 
                                            
                gate = torch.stack(logits)
                sel= self.dnn(output)
                output = torch.einsum('abcd, bca -> bcd', gate, sel)            

                out = output[:, :self.viewer_number, :]

                return torch.nn.functional.normalize(out, p=2, dim=-1)
            
            else:     
                output = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]                
                out = output[:, :self.viewer_number, :]
                return torch.nn.functional.normalize(out, p=2, dim=-1)        
            '''            
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            output = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            
            return torch.nn.functional.normalize(output, p=2, dim=-1)

        else:
            if training:
                output = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]

                logits = []
                for i, dropout_op in enumerate(self.dropout_ops):
                    temp_out = dropout_op(output)
                    logits.append(temp_out) 
                                            
                gate = torch.stack(logits)
                sel= self.dnn(output)
                output = torch.einsum('abcd, bca -> bcd', gate, sel)            

                out = output[:, :self.viewer_number, :]

                return torch.nn.functional.normalize(out, p=2, dim=-1)
            
            else:     
                output = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]                
                out = output[:, :self.viewer_number, :]
                return torch.nn.functional.normalize(out, p=2, dim=-1)        
