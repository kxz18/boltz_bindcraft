#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
w1, w2 = torch.randn(33, 384), torch.randn(33, 384)
# Dummy encoder
def enc(res_type):
    s_inputs = torch.tanh(res_type @ w1)
    s = torch.sigmoid(res_type @ w2)
    N = res_type.shape[1]
    z = torch.randn(1, N, N, 128, requires_grad=True)
    d = torch.randn(1, N, N, 1, 64, requires_grad=True)
    return s_inputs, s, z, d

# Dummy iterative function
def f(res_type, s_inputs, s, z, d, x_prev):
    offset = res_type.mean() + s_inputs.mean() + s.mean() + z.mean() + d.mean()
    return x_prev + 0.01 * torch.tanh(x_prev + offset)

# Dummy readout
def o(s_inputs, s, z, d, x):
    offset = s.mean() + z.mean() + d.mean()
    return (x ** 2).mean() + s_inputs.mean() * 0.01 + offset

# Shapes
N, M, T = 8, 16, 5
res_type = torch.randn(1, N, 33, requires_grad=True)
x = torch.randn(1, M, 3, requires_grad=True)
x_init = x.clone()

# --------- baseline full autograd ---------
# Forward
s_inputs, s, z, d = enc(res_type)
for t in range(T):
    x = f(res_type, s_inputs, s, z, d, x)
L = o(s_inputs, s, z, d, x)

# Reference gradient
L.backward()
grad_ref = res_type.grad.clone()
print(f'Full graph bp: {grad_ref.shape}, {grad_ref[0][0][:10]}')

# --------- manual adjoint (memory-efficient) ---------
# reset
res_type.grad = None
x = x_init
x.grad = None
# Forward
with torch.no_grad():
    s_inputs, s, z, d = enc(res_type)
    xs = [x]
    for t in range(T):
        x = f(res_type, s_inputs, s, z, d, x)
        xs.append(x)
x.requires_grad_(True)
s_inputs.requires_grad_(True)
s.requires_grad_(True)
z.requires_grad_(True)
d.requires_grad_(True)
L = o(s_inputs, s, z, d, x)
# Manual adjoint
a, ds_inputs, ds, dz, dd = torch.autograd.grad(L, [x, s_inputs, s, z, d], retain_graph=False)
dres_type = torch.zeros_like(res_type, requires_grad=False)
# ds_inputs = torch.zeros_like(s_inputs)
# ds = torch.zeros_like(s)
# dz = torch.zeros_like(z)
# dd = torch.zeros_like(d)

for t in reversed(range(T)):
    x_t = xs[t].detach().requires_grad_(True)
    x_t_next = f(res_type, s_inputs, s, z, d, x_t)
    a, v_s_inputs, v_s, v_z, v_d, v_res_type = torch.autograd.grad(
        outputs=x_t_next,
        inputs=[x_t, s_inputs, s, z, d, res_type],
        grad_outputs=a,
        retain_graph=False,
        allow_unused=True,
    )
    with torch.no_grad():
        if v_s_inputs is not None: ds_inputs += v_s_inputs
        if v_s is not None: ds += v_s
        if v_z is not None: dz += v_z
        if v_d is not None: dd += v_d
        if v_res_type is not None: dres_type += v_res_type

# Chain rule
s_inputs, s, z, d = enc(res_type)
dres_type += torch.autograd.grad(
    outputs=[s_inputs, s, z, d],
    inputs=res_type,
    grad_outputs=[ds_inputs, ds, dz, dd],
)[0]
print(f'Adjoint bp: {dres_type.shape}, {dres_type[0][0][:10]}')

print("Adjoint vs autograd diff:",
      (dres_type - grad_ref).abs().max().item())
