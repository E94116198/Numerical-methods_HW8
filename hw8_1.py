import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# a. 二次多項式擬合 y = a2*x^2 + a1*x + a0
coeffs_poly2 = np.polyfit(x, y, deg=2)
y_poly2 = np.polyval(coeffs_poly2, x)
error_poly2 = np.sum((y - y_poly2) ** 2)
a2, a1, a0 = coeffs_poly2

# b. 指數擬合 y = b * e^(a*x)
log_y = np.log(y)
reg_exp = LinearRegression().fit(x.reshape(-1, 1), log_y)
a_exp = reg_exp.coef_[0]
ln_b_exp = reg_exp.intercept_
b_exp = np.exp(ln_b_exp)
y_exp = b_exp * np.exp(a_exp * x)
error_exp = np.sum((y - y_exp) ** 2)

# c. 冪次擬合 y = b * x^n
log_x = np.log(x)
reg_pow = LinearRegression().fit(log_x.reshape(-1, 1), log_y)
n_pow = reg_pow.coef_[0]
ln_b_pow = reg_pow.intercept_
b_pow = np.exp(ln_b_pow)
y_pow = b_pow * x**n_pow
error_pow = np.sum((y - y_pow) ** 2)

# 輸出結果
print("a. Polynomial Fit (degree 2) y = a2 * x^2 + a1 * x + a0:")
print(f"   y = {a2:.4f}x^2 + {a1:.4f}x + {a0:.4f}")
print(f"   Error = {error_poly2:.5f}\n")

print("b. Exponential Fit y = b * e^(a*x) :")
print(f"   y = {b_exp:.4f} * e^({a_exp:.4f}x)")
print(f"   Error = {error_exp:.5f}\n")

print("c. Power Fit y = b * x^n :")
print(f"   y = {b_pow:.4f} * x^{n_pow:.4f}")
print(f"   Error = {error_pow:.5f}\n")
