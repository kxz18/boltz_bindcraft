import time
import torch

# -------------------
# Define the forward map
# -------------------
def f(X, s):
    # X: (batch, dim)
    # s: scalar tensor, (batch, 1)
    return X + s * torch.tanh(X)

# -------------------
# Define the loss
# -------------------
def loss_fn(X0, X_target):
    return ((X0 - X_target)**2).mean()

# -------------------
# Simple test
# -------------------
torch.manual_seed(0)
dim = 4
T = 5
batch = 3

# s = torch.tensor(0.5, requires_grad=True)
s = torch.randn(batch, 1, requires_grad=True)
X_T = torch.randn(batch, dim)
X_target = torch.zeros_like(X_T)

# --------- baseline full autograd ---------
X = X_T.clone()
start = time.time()
for t in range(T, 0, -1):
    X = f(X, s)
L_full = loss_fn(X, X_target)
L_full.backward()
print(f'Elapsed {time.time() - start}s')
grad_full = s.grad

print('Autograd (full-graph) gradient:', grad_full)

# --------- manual adjoint (memory-efficient) ---------
s.grad = None  # reset
X = X_T.clone()
Xs = [X_T.clone()]

# forward pass (store only X_t's)
start = time.time()
with torch.no_grad():
    for t in range(T, 0, -1):
        X = f(X, s)
        Xs.append(X.clone())  # Xs[t] = X_{t-1}
print(f'Forward elapsed: {time.time() - start}s')

X0 = Xs[-1].detach().requires_grad_(True)
L = loss_fn(X0, X_target)

# compute dL/dX0
a = torch.autograd.grad(L, X0, retain_graph=False)[0]
ds = torch.zeros_like(s)

# reverse loop
for t in range(T, 0, -1):
    X_t = Xs[t-1].detach().requires_grad_(True) # actually it's looping from zero to T-1
    X_t_minus_1 = f(X_t, s)
    v_s, a = torch.autograd.grad(X_t_minus_1, [s, X_t], grad_outputs=a, retain_graph=False)
    with torch.no_grad():
        if v_s is not None: ds = ds + v_s
    # a = torch.autograd.grad(X_t_minus_1, X_t, grad_outputs=a, retain_graph=False)[0]

print(f'Elapsed {time.time() - start}s')
print("Manual adjoint gradient:", ds)
