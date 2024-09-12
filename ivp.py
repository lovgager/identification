import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm
# _ = np.seterr(all='raise')

#%% init

N = 40
dt = 1/N
t_full = np.linspace(0, 1, N + 1)
t = t_full[:-1]
theta = 0.3
c = 1/theta
np.random.seed(1)

z_exact = 2 - np.exp(-3*t_full)
u_exact = np.empty(t.shape)
u_exact = 3*np.exp(-3*t) + c*np.sqrt(((2 - np.exp(-3*t))/(1 - t))**2 - 1)
z_emp = z_exact + np.random.normal(scale=0.05, size=z_exact.shape)
z_emp[0] = 1

#%% naive

# for n in range(N):
#     u[n] = (z_emp[n+1] - z_emp[n])/dt + c*np.sqrt((z_emp[n]/(1 - t[n]))**2 - 1)

#%% curve fitting
    
# u_func = lambda t, a, b, c, d: np.exp(a*t + b) + c*t**2 + d*t
# popt, pcov = curve_fit(u_func, t, u)

#%% recovering z

# f = lambda t, z: -c*np.sqrt((z/(1 - t))**2 - 1) + u_func(t, *popt)
# sol = solve_ivp(f, [0, t[-1]], [1], t_eval=t)

#%% plots z

# plt.figure(dpi=150)
# plt.plot(t_full, z_exact)
# plt.plot(t_full, z_emp, '.-')
# plt.xlabel('t')
# plt.plot(sol.t, sol.y[0], '.-')
# plt.grid()
# plt.legend(['z(t) точная', 'z(t) эмпирическая', 'z(t) восстановленная'])

#%% plots u

# plt.figure(dpi=150)
# plt.plot(t, u_exact)
# plt.plot(t, u, '.-', color='g')
# plt.plot(t, u_func(t, *popt), color='r')
# plt.grid()
# plt.xlabel('t')
# plt.title('"Наивный" способ восстановления u(t)')
# plt.legend(['u_n', 'МНК'])
# #plt.legend(['u(t) exact', 'u(t) naive', 'least squares fit'])

#%% Runge-Kutta 2

# def recover_z(u: np.ndarray):
#     z_recov = np.empty(t_full.shape)
#     z_recov[0] = 1
#     u_func = lambda t: u[int(t/dt)]
#     f = lambda t, z: -c*np.sqrt((z/(1 - t))**2 - 1) + u_func(t)
#     for n in range(N):
#         z_tmp = z_recov[n] + dt/2*f(t[n], z_recov[n])
#         z_recov[n+1] = z_recov[n] + dt*f(t[n] + dt/2, z_tmp)
#     return z_recov

# z_rec = recover_z(u)

#%% Runge-Kutta 4

# def recover_z4(u: np.ndarray):
#     z = np.empty(t.shape)
#     z[0] = 1
#     u_d = np.empty(N*2)
#     for n in range(N):
#         u_d[2*n] = u[n]
#     for n in range(N-1):
#         u_d[2*n + 1] = 0.5*(u_d[2*n] + u_d[2*n + 2])
#     u_d[2*N - 1] = 2*u_d[2*N - 2] - u_d[2*N - 3]
#     uf = lambda t: u_d[int(2*t/dt)]
#     f = lambda t, z: -c*np.sqrt(np.max([(z/(1 - t))**2 - 1, 0])) + uf(t)
#     for n in range(N-1):
#         k1 = f(t[n], z[n])
#         k2 = f(t[n] + dt/2, z[n] + k1*dt/2)
#         k3 = f(t[n] + dt/2, z[n] + k2*dt/2)
#         k4 = f(t[n] + dt, z[n] + dt*k3)
#         z[n+1] = z[n] + (k1 + 2*k2 + 2*k3 + k4)*dt/6
#     return z

# u = np.empty(N)
# for n in range(N):
#     u[n] = (z_emp[n+1] - z_emp[n])/dt + c*np.sqrt((z_emp[n]/(1 - t[n]))**2 - 1)

# z_rec = recover_z4(u)

#%% curve fitting z

# z_func = lambda t, a, b, c, d, e, f: a*t**5 + b*t**4 + c*t**3 + d*t**2 + e*t + f
# popt, pcov = curve_fit(z_func, t_full, z_emp)
# tnew = np.linspace(0, 1, 1000)
# z_emp_smooth = z_func(t_full, *popt)

# plt.figure(dpi=150)
# plt.plot(tnew, z_func(tnew, *popt))
# plt.plot(t_full, z_emp, '.-')
# plt.grid()
# plt.xlabel('t')
# plt.title('Сглаживание эмпирических данных z')
# plt.legend(['МНК', 'z эмпирическая'])

#%% minimize

# u = np.empty(N)
# for n in range(N):
#     u[n] = (z_emp[n+1] - z_emp[n])/dt + c*np.sqrt(np.abs((z_emp[n]/(1 - t[n]))**2 - 1))

# def to_min(u: np.ndarray):
#     z_rec = recover_z4(u)
#     return np.linalg.norm(z_emp_smooth[:-1] - z_rec)

# res = minimize(to_min, u, method='nelder-mead', options={'disp': True, 'maxiter': 1000})
# u_rec = res.x
# z_rec = recover_z4(u_rec)

#%% plots z 2

# u_func = lambda t, a, b, c0, c1, c2, c3, c4, c5: \
#     np.exp(a*t + b) + c5*t**5 + c4*t**4 + c3*t**3 + c2*t**2 + c1*t + c0
# popt, pcov = curve_fit(u_func, t, u_rec)
# u_rec_smooth = u_func(t, *popt)
# z_rec_u_smooth = recover_z4(u_rec_smooth)

# z_func = lambda t, a, b, c0, c1, c2, c3, c4, c5: \
#     np.exp(a*t + b) + c5*t**5 + c4*t**4 + c3*t**3 + c2*t**2 + c1*t + c0
# popt, pcov = curve_fit(z_func, t, z_rec_u_smooth)
# tnew = np.linspace(0, 1, 1000)
# z_rec_smooth = z_func(t, *popt)

# plt.figure(dpi=150)
# plt.plot(t_full, z_exact)
# plt.plot(t, z_rec, '.-')
# plt.grid()
# plt.legend(['z(t) точная', 
#             'z(t) восстановленная'])

#%% plots errors

# err_emp_short = np.abs(z_emp_smooth[:-1] - z_rec_smooth)
# err_exact_short = np.abs(z_exact[:-1] - z_rec_smooth)
# plt.plot(t, err_emp_short, '.-')
# plt.plot(t, err_exact_short, '.-')
# plt.grid()
# plt.legend(['error empirical', 'error exact'])

#%% curve fitting u

# plt.figure(dpi=150)
# plt.plot(t, u_exact, color='r')
# plt.plot(t, u_rec, '.-', color='g')
# plt.grid()
# plt.legend(['u(t) точная', 'u(t) восстановленная'])

#%% solve function

def recover_z(u: np.ndarray):
    u_d = np.empty(N*2)
    for n in range(N):
        u_d[2*n] = u[n]
    for n in range(N-1):
        u_d[2*n + 1] = 0.5*(u_d[2*n] + u_d[2*n + 2])
    u_d[2*N - 1] = 2*u_d[2*N - 2] - u_d[2*N - 3]
    uf = lambda t: u_d[int(2*t/dt)]
    s = lambda t, z: (np.abs(z**2/(1 - t)**2 - 1) + z**2/(1 - t)**2 - 1)/2
    f = lambda t, z: -c*np.sqrt(s(t, z)) + uf(t)
    res = solve_ivp(f, [0, 1-dt], [1], t_eval=np.arange(0, 1-dt, dt), method='LSODA')
    return res.y

#%% read y

ydata = np.asarray(pd.read_excel('data.xlsx', header=2, usecols='B,C')['y'])
ydata[0] = 1

psi = 2
N = 40
dt = 1/N
t_full = np.linspace(0, 1, N + 1)
t = t_full[:-1]
theta = 0.3
c = 1/theta

plt.figure(dpi=150)
plt.plot(t_full, ydata, '.-')
plt.grid()
plt.legend(['y'])

#%% plot initial z

zdata = ydata*(1 - t_full)

# z_right = 10/9*(1 - t_full[4:])
# z_left = np.ones(4)
# z_piece = np.concatenate([z_left, z_right])

plt.figure(dpi=150)
plt.plot(t_full, zdata, '.-', c='purple')
# plt.plot(t_full, z_piece)
plt.grid()
plt.legend(['z'])

#%% u naive

u = np.zeros(N)
for n in range(N):
    u[n] = (zdata[n+1] - zdata[n])/dt + c*np.sqrt((zdata[n]/(1 - t[n]))**2 - 1)

#%% average

def avg(u: np.ndarray, k=1):
    N = len(u)
    avg = np.empty(N)
    for i in range(k):
        avg[i] = np.mean(u[:(i+1)])
    for i in range(k, N):
        avg[i] = np.mean(u[(i-k):(i+1)])
    u_avg = np.empty(N)
    u_avg[:(-k)] = avg[k:]
    u_avg[-k:] = u[-k:]
    return u_avg

u_avg = avg(u, 1)
plt.plot(t, u)
plt.plot(t, u_avg)

#%% smooth

# u_func = lambda t, a, b, c, d, e, f: \
#     a*t**5 + b*t**4 + c*t**3 + d*t**2 + e*t + f
# popt, _ = curve_fit(u_func, t, u)
# u_smooth = u_func(t_full, *popt)

# Bernsteim polynomial
from scipy.special import comb
a = [ydata[k]*comb(N, k)*t**k*(1 - t)**(N-k) \
     for k in range(N+1)]
u_smooth = np.sum(a, axis=0)

plt.figure(dpi=150)
plt.plot(t, u, '.-', c='C2')
plt.plot(t_full, u_smooth, c='C3')
plt.grid()
plt.legend(['u_n', 'МНК'])
_ = plt.title('"Наивный" способ восстановления u(t)')

#%% coordinate descent

def to_min(u: np.ndarray):
    z_rec = recover_z(u)[0]
    y_rec = z_rec/(1 - t[:-1])
    # return np.linalg.norm(z[:-2] - z_rec, ord=np.infty)
    return np.linalg.norm(ydata[:-2] - y_rec, ord=np.infty)


def coordinate_descent(func, initial_guess, tolerance=1e-6, max_iterations=100, learning_rate=0.5):
    params = initial_guess.copy()
    prev_loss = func(params)
    num_params = len(params)
    for it in tqdm(range(max_iterations)):
        for i in range(num_params):
            params_try = params.copy()
            params_try[i] += learning_rate
            loss_try = func(params_try)
            loss_cur = func(params)
            if loss_try < loss_cur:
                params[i] = params_try[i]
            else:
                params_try[i] -= 2 * learning_rate
                loss_try = func(params_try)
                if loss_try < loss_cur:
                    params[i] = params_try[i]
        if np.abs(prev_loss - func(params)) < tolerance:
            learning_rate *= 0.5
        prev_loss = func(params)
        print(prev_loss)
    return params

initial_guess = avg(u, 1)
u_opt = coordinate_descent(to_min, initial_guess)
z_opt = recover_z(u_opt)[0]
y_opt = z_opt/(1 - t[:-1])
z_err = np.linalg.norm(z_opt - zdata[:-2], ord=np.infty)
y_err = np.linalg.norm(y_opt - ydata[:-2], ord=np.infty)
print(f'z err: {z_err}')
print(f'y err: {y_err}')



#%% second try

u_opt2 = coordinate_descent(to_min, avg(u_opt, 1), learning_rate=0.1)
z_opt2 = recover_z(u_opt2)[0]
y_opt2 = z_opt2/(1 - t[:-1])
z_err2 = np.linalg.norm(z_opt2 - zdata[:-2], ord=np.infty)
y_err2 = np.linalg.norm(y_opt2 - ydata[:-2], ord=np.infty)
print(f'z err2: {z_err2}')
print(f'y err2: {y_err2}')


#%% plots u

plt.figure(dpi=150)
# plt.plot(t, u, '.-')
plt.plot(t, u_opt, '.-', c='C2')
plt.grid()
plt.legend(['u(t) оптимальная'])

print(to_min(u))
print(to_min(u_opt))


#%% plot z

plt.plot(t_full, zdata, '.-', c='purple')
plt.plot(t[:-1], z_opt, '.-', c='r')
plt.grid()
plt.legend(['z(t) эмпирическая', 
            'z(t) восст. коорд. спуском'])

#%% return to y

y_naive = recover_z(u_smooth)[0]/(1 - t[:-1])

plt.figure(dpi=150)
plt.plot(t, ydata[:-1], '.-')
plt.plot(t[:-1], y_naive, '.-')
plt.plot(t[:-1], y_opt, '.-')
plt.grid()
plt.legend(['y(t) эмпирическая (исходные данные)', 
            'y(t) наивная (1-й способ)', 
            'y(t) оптимальная (2-й способ)'])

#%% export

f_opt = u_opt/psi
# pd.DataFrame(f_opt).to_excel('result.xlsx')
plt.figure(dpi=150)
plt.grid()
plt.plot(t, f_opt, '.-', c='C2')
plt.legend(['f(t)'])
