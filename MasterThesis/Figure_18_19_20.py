# This Python code was written by Tomas Arzola Röber
# on May 9th, 2023.
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.integrate import odeint

#Parameter values
H = 1.3
kappa = 0.07
xi = 0.1
pD = 0.038
alpha = 0.26
r = 0.5
q = 2
m = 1
delta = 0.03
v = 0.085
b = 40
s = 1

# P-Nullcline
def nullcline_p(P, pD=pD, alpha=alpha, r=r, q=q, m=m, delta=delta):
    return (pD - alpha * P + (r * np.power(P, q) / (np.power(m, q) + np.power(P, q)))) / delta

#non-trivial F-Nullcline
def nullcline_f(P, H=H, kappa=kappa, v=v, b=b):
    return -(kappa * np.power(P, b) / (xi * (np.power(H, b) + np.power(P, b)))) + v / xi

#ODE System
def DGL(z, t, H=H, xi=xi, pD=pD, alpha=alpha, r=r, q=q, m=m, delta=delta, kappa=kappa, v=v, b=b, s=s):
    P, F = z
    dzdt = [pD - delta * F - alpha * P + (r * np.power(P, q) / (np.power(m, q) + np.power(P, q))),
            s * F * (1 - F) * (-v + xi * F + (kappa * np.power(P, b) / (np.power(H, b) + np.power(P, b))))]
    return dzdt


def dPdt(P, F, pD=pD, delta=delta, alpha=alpha, r=r, q=q, m=m):
    return pD - delta * F - alpha * P + (r * np.power(P, q) / (np.power(m, q) + np.power(P, q)))


def dFdt(P, F, H=H, kappa=kappa, v=v, b=b, s=s, xi=xi):
    return s * F * (1 - F) * (-v + xi * F + (kappa * np.power(P, b) / (np.power(H, b) + np.power(P, b))))

#Stability with Jacobian Matrix and eigenvalues
def stability(P, F, H=H, xi=xi, pD=pD, alpha=alpha, r=r, q=q, m=m, delta=delta, kappa=kappa, v=v, b=b, s=s):
    x = sym.Symbol('x')

    ax = sym.diff(dPdt(x, F, pD=pD, delta=delta, alpha=alpha, r=r, q=q, m=m), x)
    ddx = sym.lambdify(x, ax)
    a = ddx(P)

    bx = sym.diff(dPdt(P, x, pD=pD, delta=delta, alpha=alpha, r=r, q=q, m=m), x)
    ddx = sym.lambdify(x, bx)
    be = ddx(F)

    cx = sym.diff(dFdt(x, F, H=H, kappa=kappa, v=v, b=b, s=s, xi=xi), x)
    ddx = sym.lambdify(x, cx)
    c = ddx(P)

    dx = sym.diff(dFdt(P, x, H=H, kappa=kappa, v=v, b=b, s=s, xi=xi), x)
    ddx = sym.lambdify(x, dx)
    d = ddx(F)

    matrix = [[a, be], [c, d]]

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    e1 = eigenvalues[0]
    e2 = eigenvalues[1]

    if e1 >= 0 or e2 >= 0:
        return 0, e1, e2
    else:
        return 1, e1, e2


p = np.linspace(0, 2, 100000)
one_nullcline = np.ones_like(p)
zero_nullcline = np.zeros_like(p)

null_P = np.array(nullcline_p(p))
null_F = np.array(nullcline_f(p))

fig, ax = plt.subplots()
ax.plot(p, null_F, color="blue", label="F-nullcline")
ax.plot(p, null_P, color="green", label="P-nullcline")
ax.plot(p, one_nullcline, color="blue")
ax.plot(p, zero_nullcline, color="blue")

idx = np.argwhere(np.diff(np.sign(null_F - null_P))).flatten()
idx1 = np.argwhere(np.diff(np.sign(one_nullcline - null_P))).flatten()
idx0 = np.argwhere(np.diff(np.sign(null_P - zero_nullcline))).flatten()

#Plot and stability
print("Non-trivial equilibrium:\n")
for i in idx:
    if 1 >= null_F[i] >= 0:
        print("P: " + str(p[i]) + ", F: " + str(null_P[i]))
        if stability(p[i], null_P[i])[0]:
            print("Stable")
            print("Eigenvalues: " + str(stability(p[i], null_P[i])[1]) + " und " + str(stability(p[i], null_P[i])[2]))
            ax.plot(p[i], null_P[i], 'ko')
        else:
            print("Unstable")
            print("Eigenvalues: " + str(stability(p[i], null_P[i])[1]) + " und " + str(stability(p[i], null_P[i])[2]))
            ax.plot(p[i], null_P[i], 'o', markerfacecolor="white", markeredgecolor="black")

print("\nFull cooperation equilibrium:\n ")
for i in idx1:
    print("P: " + str(p[i]) + ", F: " + str(1))
    if stability(p[i], 1)[0]:
        print("Stable")
        print("Eigenvalues: " + str(stability(p[i], null_P[i])[1]) + " und " + str(stability(p[i], null_P[i])[2]))
        ax.plot(p[i], 1, 'ko')
    else:
        print("Unstable")
        print("Eigenvalues: " + str(stability(p[i], null_P[i])[1]) + " und " + str(stability(p[i], null_P[i])[2]))
        ax.plot(p[i], 1, 'o', markerfacecolor="white", markeredgecolor="black")

print("\nFull defection equilibrium:\n ")
for i in idx0:
    print("P: " + str(p[i]) + ", F: " + str(0))
    if stability(p[i], 0)[0]:
        print("Stable")
        print("Eigenvalues: " + str(stability(p[i], null_P[i])[1]) + " und " + str(stability(p[i], null_P[i])[2]))
        ax.plot(p[i], 0, 'ko')
    else:
        print("Unstable")
        print("Eigenvalues: " + str(stability(p[i], null_P[i])[1]) + " und " + str(stability(p[i], null_P[i])[2]))
        ax.plot(p[i], 0, 'o', markerfacecolor="white", markeredgecolor="black")

x_range = np.linspace(-0.02, 2.02, 10000)
y_range = np.linspace(-0.02, 2.02, 10000)

x = np.linspace(-0.02, 2.02, 30)
y = np.linspace(-0.02, 2.02, 30)

t = np.arange(0, 2000, 0.1)
sol_dgl = odeint(DGL, [1.350, 0.319], t)
ax.plot(sol_dgl[:, 0], sol_dgl[:, 1], color="black")
ax.plot(sol_dgl[:, 0][0], sol_dgl[:, 1][0], 'x', color="black", label="Initial Condition")
ax.set(xlabel='P', ylabel='F')
ax.set_xlim(0, 1.5)
ax.set_ylim(-0.01, 1.01)
plt.show()

fig, ax = plt.subplots()
ax.plot(t, sol_dgl[:, 0], color="green", label="Phosphorus Concentration")
ax.plot(t, sol_dgl[:, 1], color="blue", label="Fraction of cooperators")
ax.set(xlabel='time', ylabel='P, F')
plt.show()