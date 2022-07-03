# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class ToepliztV2(nn.Module):
    def __init__(self, n, type_num=-1, causal=False):
        super().__init__()
        self.type_num = type_num
        self.n = n
        self.causal = causal
        self.infty = -1e20
        if self.type_num == -1:
            # [1,...,n-1]
            self.pos = nn.Parameter(torch.ones(n - 1))
            # [0]
            self.zero = nn.Parameter(torch.ones(1))
            # [-(n-1),...,-1]
            if self.causal:
                self.neg = nn.Parameter(torch.ones(n - 1) * self.infty, requires_grad=False)
            else:
                self.neg = nn.Parameter(torch.ones(n - 1))
            # # [1,...,n-1]
            # self.pos = nn.Parameter(torch.arange(1, n).float())
            # # [0]
            # self.zero = nn.Parameter(torch.zeros(1))
            # # [-(n-1),...,-1]
            # self.neg = nn.Parameter(-torch.arange(n, 0, -1).float())
        elif self.type_num == 1:
            # exp(-|i - j|)
            # [1,...,n-1]
            vec = torch.clamp(-torch.ones(n - 1).float(), max=30, min=-60)
            self.pos = nn.Parameter(vec, requires_grad=False)
            # [0]
            self.zero = nn.Parameter(torch.ones(1), requires_grad=False)
            # [-(n-1),...,-1]
            vec = torch.clamp(-torch.arange(n, 0, -1).float(), max=30, min=-60)
            if self.causal:
                self.neg = nn.Parameter(torch.ones(n - 1) * self.infty, requires_grad=False)
            else:
                self.neg = nn.Parameter(vec, requires_grad=False)

        # print(self.toeplizt_matrix(10))

    def forward(self, x, dim=1, normalize=False, use_exp=True):
        # shape of x: b, n, e
        b = x.shape[0]
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        
        pos = torch.cat([self.pos[:l1], torch.ones(l2).to(x) * self.infty])
        neg = torch.cat([torch.ones(l2).to(x) * self.infty, self.neg[-l1:]])
        if use_exp:
            a = torch.exp(torch.clamp(torch.cat([self.zero, pos, self.zero, neg]), max=30, min=-60))
        else:
            a = torch.cat([self.zero, pos, self.zero, neg])
        output = self.compute(x, a, dim, n)

        if normalize:
            ones = torch.ones(b, n, 1).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
        ##### for test
        # matrix = self.toeplizt_matrix(n)
        # res = torch.einsum('nm,bme->bne', matrix, x)
        # print(torch.norm(res - output))
        # ones = torch.ones(b, n, 1).to(x)
        # denorm = self.compute(ones, a, dim, n)
        # print(torch.sum(matrix / denorm, dim=-1))
        ##### for test
    
    def compute(self, x, a, dim, n):
        y = torch.fft.fft(x, 2 * n, dim=dim, norm="ortho")
        v = torch.fft.fft(a).unsqueeze(0).unsqueeze(-1)
        u = v * y
        output = torch.fft.ifft(u, 2 * n, dim=dim, norm="ortho")[:, :n, :].real

        return output

    def toeplizt_matrix(self, n):
        # c: first col, r: first row
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        pos = torch.clamp(torch.cat([self.pos[:l1], torch.ones(l2) * self.infty]), max=30, min=-60)
        neg = torch.clamp(torch.cat([torch.ones(l2) * self.infty, self.neg[-l1:]]), max=30, min=-60)
        c = torch.exp(torch.cat([self.zero, pos]))
        r = torch.exp(torch.cat([self.zero, neg.flip(0)]))
        vals = torch.cat([r, c[1:].flip(0)])
        shape = len(c), len(r)
        i, j = torch.ones(*shape).nonzero().T
        res = vals[j - i].reshape(*shape)

        return res


# model = ToepliztV2(100)
# b = 1
# n = 200
# e = 4
# x = torch.rand(b, n, e)
# print(x)
# print(model(x))