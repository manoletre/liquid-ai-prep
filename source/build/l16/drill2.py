import numpy as np

def forward_backward(x, w1, w2, y):
    z = w1 * x; h = max(z, 0.0)          # ReLU
    yhat = w2 * h
    L = 0.5 * (yhat - y) ** 2
    d_yhat = yhat - y
    g_w2 = d_yhat * h
    d_h = d_yhat * w2
    d_z = d_h * (1.0 if z > 0 else 0.0)  # ReLU gate
    g_w1 = d_z * x
    return L, g_w1, g_w2

L, g1, g2 = forward_backward(2.0, 2.0, 0.5, 3.0)
print(L, g1, g2)
eps = 1e-6
num_g1 = (forward_backward(2.0, 2.0 + eps, 0.5, 3.0)[0] - forward_backward(2.0, 2.0 - eps, 0.5, 3.0)[0]) / (2 * eps)
print("finite diff w1:", round(num_g1, 6))
print(forward_backward(1.0, 3.0, -1.0, 0.0))
print(forward_backward(2.0, -1.0, 0.5, 3.0))
