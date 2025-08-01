#!/usr/bin/env python3
#author:rangapv@yahoo.com
#01-08-25

import time

import mlx.core as mx

num_features = 100
num_examples = 1_000
num_iters = 10_000
lr = 0.01

# True parameters
w_star = mx.random.normal((num_features,))
lenofw = len(w_star)

print(f'w-star is {w_star}')

print(f'the length of w star is {lenofw}')

# Input examples (design matrix)
X = mx.random.normal((num_examples, num_features))
lenofX = len(X)
lenofX1 = len(X[0])

print(f'design matrix is {X}')

print(f'length of matrix is {lenofX}')
print(f'length of matrix X1 is {lenofX1}')

# Noisy labels
eps = 1e-2 * mx.random.normal((num_examples,))
epslen = len(eps)


print(f'the labels are {eps}')
print(f'the length of labels are {epslen}')

y = X @ w_star + eps

# Initialize random parameters
w = 1e-2 * mx.random.normal((num_features,))
lenofw = len(w)

print(f'w is {w}')
print(f'lenofw is {lenofw}')

def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))

grad_fn = mx.grad(loss_fn)

tic = time.time()
for _ in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad
    mx.eval(w)
toc = time.time()

loss = loss_fn(w)
error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5
throughput = num_iters / (toc - tic)

print(
    f"Loss {loss.item():.5f}, L2 distance: |w-w*| = {error_norm:.5f}, "
    f"Throughput {throughput:.5f} (it/s)"
)
