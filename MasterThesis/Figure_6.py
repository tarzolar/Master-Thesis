# This Python code was written by Tomas Arzola Röber
# on May 9th, 2023.
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.integrate import odeint

#Parameter values
H = 0.1
kappa = 0.05
xi = 0.025
pD = 0.04
alpha = 0.26
r = 0.52
q = 2
m = 1
delta = 0.035
v = 0.041
b = 2
s = 1

#P-Nullcline
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

# Stability with Jacobian Matrix and Eigenvalues
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

# Plot of the Modell and its stability
count = 0
fig, ax = plt.subplots(2, 3, figsize=(20, 20))
h = [0.02, 0.1, 0.1687, 0.5651, 1.533, 1.798]
for i in range(2):
    for j in range(3):
        p = np.linspace(0, 2, 100000)
        one_nullcline = np.ones_like(p)
        zero_nullcline = np.zeros_like(p)
        null_P = np.array(nullcline_p(p))
        null_F = np.array(nullcline_f(p, H=h[count]))
        ax[i, j].plot(p, null_F, color="blue", label="F-nullcline")
        ax[i, j].plot(p, null_P, color="green", label="P-nullcline")
        ax[i, j].plot(p, one_nullcline, color="blue")
        ax[i, j].plot(p, zero_nullcline, color="blue")

        idx = np.argwhere(np.diff(np.sign(null_F - null_P))).flatten()
        idx1 = np.argwhere(np.diff(np.sign(one_nullcline - null_P))).flatten()
        idx0 = np.argwhere(np.diff(np.sign(null_P - zero_nullcline))).flatten()


        print("Non-trivial equilibrium:\n")
        for a in idx:
            if 1 >= null_F[a] >= 0:
                print("P: " + str(p[a]) + ", F: " + str(null_P[a]))
                if stability(p[a], null_P[a], H=h[count])[0]:
                    print("Stable")
                    print("Eigenvalues: " + str(stability(p[a], null_P[a], H=h[count])[1]) + " und " + str(
                        stability(p[a], null_P[a], H=h[count])[2]))
                    ax[i, j].plot(p[a], null_P[a], 'ko', markersize=10)
                else:
                    print("Unstable")
                    print("Eigenvalues: " + str(stability(p[a], null_P[a], H=h[count])[1]) + " und " + str(
                        stability(p[a], null_P[a], H=h[count])[2]))
                    ax[i, j].plot(p[a], null_P[a], 'o', markerfacecolor="white", markeredgecolor="black", markersize=10)

        print("\nFull cooperation equilibrium:\n ")
        for a in idx1:
            print("P: " + str(p[a]) + ", F: " + str(1))
            if stability(p[a], 1, H=h[count])[0]:
                print("Stable")
                print("Eigenvalues: " + str(stability(p[a], null_P[a], H=h[count])[1]) + " und " + str(
                    stability(p[a], null_P[a], H=h[count])[2]))
                ax[i, j].plot(p[a], 1, 'ko', markersize=10)
            else:
                print("Unstable")
                print("Eigenvalues: " + str(stability(p[a], null_P[a], H=h[count])[1]) + " und " + str(
                    stability(p[a], null_P[a], H=h[count])[2]))
                ax[i, j].plot(p[a], 1, 'o', markerfacecolor="white", markeredgecolor="black", markersize=10)

        print("\nFull defection equilibrium:\n ")
        for a in idx0:
            print("P: " + str(p[a]) + ", F: " + str(0))
            if stability(p[a], 0, H=h[count])[0]:
                print("Stable")
                print("Eigenvalues: " + str(stability(p[a], null_P[a], H=h[count])[1]) + " und " + str(
                    stability(p[a], null_P[a], H=h[count])[2]))
                ax[i, j].plot(p[a], 0, 'ko', markersize=10)
            else:
                print("Unstable")
                print("Eigenvalues: " + str(stability(p[a], null_P[a], H=h[count])[1]) + " und " + str(
                    stability(p[a], null_P[a], H=h[count])[2]))
                ax[i, j].plot(p[a], 0, 'o', markerfacecolor="white", markeredgecolor="black", markersize=10)

        ax[i, j].set_xlim(0, 1.6)
        ax[i, j].set_ylim(-0.01, 1.01)
        ax[i, j].set_title("H = " + str(h[count]), fontsize=15)
        count += 1

t = np.arange(0, 5000, 0.1)
sol_dgl = odeint(DGL, [0.227, 0.417], t, args=(h[2], xi, pD, alpha, r, q, m, delta, kappa, v, b, s))
ax[0, 2].plot(sol_dgl[:, 0], sol_dgl[:, 1], color="black")
ax[0, 2].plot(sol_dgl[:, 0][0], sol_dgl[:, 1][0], 'x', color="black", label="Initial Condition")
fig.supxlabel('P', fontsize=30)
fig.supylabel('F', fontsize=30)
plt.show()