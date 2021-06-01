import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def generate_data (y0 = ((1., 150., 0.)),t = np.array((0,100))):
    I = 400
    F_N =40
    u_m = 0.0572
    u_d = 0
    K_N = 393.1
    Y_NX = 504.5
    k_m = 0.00016
    k_d = 0.281
    k_s = 178.9
    k_i = 447.1
    k_sq = 23.51
    k_iq = 800
    K_Nq = 16.89
    a = u_m * I / (I + k_s+ I**2/k_i)
    b = -Y_NX * u_m * I / (I + k_s+ I**2/k_i)
    c = k_m * I / (I + k_sq+ I**2/k_iq)

    def ode3 (y,t):
        c_x, c_n, c_q = y
        dydt = [a * c_x * c_n / (c_n + K_N) - u_d * c_x, b * c_x * c_n / (c_n + K_N) + F_N, c * c_x - k_d * c_q / (c_n + K_Nq)]
        return dydt

    sol = odeint(ode3, y0, t)
    return sol
# sol = generate_data(t=np.array((0,500)))
# print (sol)
# print(sol[:,2])
# print(a),print(b),print(c)
# plt.plot(t,sol[:,0],"b",label = "c_x")
# plt.plot(t,sol[:,1],"g",label = "c_n")
# plt.plot(t,sol[:,2],"r",label = "c_q")
# plt.legend()
# plt.show()

def discrete_data (change_initial = False, time_upper = 1000, timer_lower = 0,num_point = 2000,is_plot = False):
    input = np.zeros(shape = (num_point+1,4))
    output = np.zeros(shape = (num_point+1,3))
    if change_initial == True:
        y0 = np.array((0.9 + 0.2* np.random.random_sample(), 135.0 + 30 * np.random.random_sample(), 0))
    else:
        y0 = np.array((2.0 , 170 , 0.0))
    t0 = np.array((0,))
    tf = np.linspace(timer_lower + 0.01,time_upper,num_point)
    t = np.concatenate((t0,tf),axis = 0)
    sol = generate_data(y0=y0, t = t)
    for i in range (num_point+1):
       input[i] = np.append(y0,t[i])
    output = sol

    if is_plot == True:
        plt.plot(output[:,0])
        plt.plot(output[:, 1])
        plt.plot(output[:, 2])
        plt.show()
    return input,output

input,output = discrete_data(time_upper=1000,timer_lower=0,is_plot=True)

np.savetxt("input_2.0_170_0_t.txt",input)
np.savetxt("output_2.0_170_0_t.txt",output)
