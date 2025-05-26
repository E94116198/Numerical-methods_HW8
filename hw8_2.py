import numpy as np
from scipy.integrate import quad
from numpy.polynomial.polynomial import Polynomial

# 函數 f(x)
def f(x):
    return 1/2 * np.cos(x) + 1/4 * np.sin(2 * x)

# 基底函數：1, x, x^2
basis = [lambda x: 1, lambda x: x, lambda x: x**2]

# 內積計算：<f, g> = ∫[a, b] f(x)g(x) dx
def inner_product(f, g, a=-1, b=1):
    integrand = lambda x: f(x) * g(x)
    result, _ = quad(integrand, a, b)
    return result

# 計算矩陣 A 和向量 b
n = 3  # 逼近多項式的階數 + 1
A = np.zeros((n, n))
b_vec = np.zeros(n)

for i in range(n):
    for j in range(n):
        A[i, j] = inner_product(basis[i], basis[j])
    b_vec[i] = inner_product(f, basis[i])

# 解線性方程組 A * c = b 得到係數 c
coeffs = np.linalg.solve(A, b_vec)

# 顯示結果
print("P(x) =", f"{coeffs[2]:.9f}x^2 + {coeffs[1]:.9f}x + {coeffs[0]:.9f}")
