import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

m = 16
n = 4  # degree of trigonometric polynomial
xj = np.linspace(0, 1, m, endpoint=False)
fj = xj**2 * np.sin(xj)

# 計算 Fourier 係數
a0 = np.sum(fj) / m
ak = np.zeros(n)
bk = np.zeros(n)

for k in range(1, n+1):
    ak[k-1] = (2/m) * np.sum(fj * np.cos(2 * np.pi * k * xj))
    bk[k-1] = (2/m) * np.sum(fj * np.sin(2 * np.pi * k * xj))

# 定義 S4(x)
def S4(x):
    result = a0
    for k in range(1, n+1):
        result += ak[k-1] * np.cos(2 * np.pi * k * x) + bk[k-1] * np.sin(2 * np.pi * k * x)
    return result

# b. 計算 ∫ S4(x) dx
integral_S4, _ = quad(S4, 0, 1)

# c. 計算 ∫ x^2*sin(x) dx
true_integral, _ = quad(lambda x: x**2 * np.sin(x), 0, 1)

# d. 計算誤差 E(S4)
S4_vals = S4(xj)
error = np.sum((fj - S4_vals)**2)

# 輸出結果
print(f"(b) ∫ S4(x) dx = {integral_S4:.6f}")
print(f"(c) ∫ x^2 sin(x) dx = {true_integral:.6f}")
print(f"(d) Error E(S4) = {error:.6e}")
