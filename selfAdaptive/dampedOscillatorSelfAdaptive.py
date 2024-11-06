import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=400)
parser.add_argument('--test_data_size', type=int, default=400)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--tf', type=int, default=50)
parser.add_argument('--tf_test', type=int, default=50)
parser.add_argument('--savePlot', type=str)
parser.add_argument('--saveModel', type=str)
parser.add_argument('--updateStrategy', type=str, choices=["True", "False"], default="True")
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

#######DATASET##########################################################
m = 1.0       # mass
c = 0.1       # damping coefficient
k = 1.0       # spring constant
x0 = 1.0      # initial displacement
v0 = 0.0      # initial velocity

y0 = torch.tensor([x0, v0], dtype=torch.float32).to(device)

t = torch.linspace(0, args.tf, args.data_size).to(device)
t_test = torch.linspace(0, args.tf_test, args.test_data_size).to(device)

#equation of motion
def f(t, y):
    x, v = y
    return torch.stack([v, (-c*v - k*x)/m])

y = odeint(f, y0, t).to(device)
test_y = odeint(f, y0, t_test).to(device)

# Extract the displacement, velocity, and acceleration from the solution
x = y[:, 0]
v = y[:, 1]
a = (-c*v - k*x)/m

x_test = test_y[:, 0]
v_test = test_y[:, 1]
a_test = (-c*v_test - k*x_test)/m

# Check the conservation laws
E = 0.5*m*v**2 + 0.5*k*x**2   # total energy
P = -c*v*x                   # power

E_cons = torch.sum(torch.abs(E - E[0])).item()/len(t)
P_cons = torch.sum(torch.abs(P - P[-1])).item()/len(t)

# Print the results
print("#############DATASET#######################################")
print("Number of points where conservation laws are not satisfied:")
print("Energy conservation:", torch.sum(torch.abs(E - E[0]) > 1e-6).item())
print("Power conservation:", torch.sum(torch.abs(P - P[-1]) > 1e-6).item())

print("Energy conservation:", E_cons)
print("Power conservation:", P_cons)
print("###########################################################")
########################################################################



class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50,50),
            nn.ELU(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

def normalizeFunc(x):
    return torch.subtract(1, torch.divide(1, torch.add(1, x)))


if __name__ == '__main__':

    tol = 1e-4

    func = ODEFunc().to(device)
    optimizer = optim.Adam(func.parameters(), lr=1e-4)

    for itr in range(1, args.niters+1): 
        pred_y = odeint(func, y0, t).to(device)

        x = y[:, 0]
        v = y[:, 1]
        a = (-c*v - k*x)/m
        E = 0.5*m*v**2 + 0.5*k*x**2 
        P = -c*v*x                 

#####################################################
        MSE_theta = torch.square(torch.subtract(pred_y, y))

        P_theta = 0 
        F_theta = torch.mean(normalizeFunc(MSE_theta))


        #inequality constraint
        v_j = torch.abs(torch.relu(-E[0] + E).to(device))
        mu_j = torch.div(torch.sum(v_j != 0), len(v_j))
        P_theta = torch.sum(v_j)

        counter = sum(list(map(lambda x: x==0, -E[0]+E[1:]))).item() #to count how many constraints violations were =0

        #equality constraint
        v_i = torch.abs(P - P[-1])
        mu_i = torch.div(torch.sum(v_i != 0), len(v_i))
        P_theta = torch.sum(v_i)

        if itr == 1:
            theta_best = list(func.parameters())
            F_best = math.inf
            P_best = math.inf
            phi_best = math.inf
            if P_theta <= tol:
                MSE_thetaf = MSE_theta
                F_thetaf = torch.mean(normalizeFunc(MSE_theta))
            else:
                MSE_thetaf = torch.multiply(MSE_theta, 10)
                F_thetaf = torch.mean(normalizeFunc(MSE_thetaf))

        if P_theta <= tol and torch.mean(MSE_theta) < torch.mean(MSE_thetaf):
            MSE_thetaf = MSE_theta
            F_thetaf = torch.mean(normalizeFunc(MSE_theta))

        if P_theta <= tol:
            phi_theta = F_theta
        else:
            phi_theta = torch.add(F_theta, torch.add(torch.mul(mu_j, torch.div(torch.sum(normalizeFunc(v_j)), len(v_j))), torch.mul(mu_i, torch.divide(torch.mean(normalizeFunc(v_i)), len(v_i)))))


        if P_theta <= tol and F_theta < F_best:
            theta_best = list(func.parameters())
            P_best = P_theta
            F_best = F_theta
            phi_best = phi_theta
        elif P_theta <= P_best:
            if P_theta < P_best:
                theta_best = list(func.parameters())
                P_best = P_theta
                F_best = F_theta
                phi_best = phi_theta
            else:
                if F_theta <= F_best:
                    theta_best = list(func.parameters())
                    P_best = P_theta
                    F_best = F_theta
                    phi_best = phi_theta
        else:
            if args.updateStrategy == "True":
                for prev, upd in zip(theta_best, theta):
                    upd.data = prev.data

#####################################################
        if itr != args.niters:
            optimizer.zero_grad()
            phi_theta.backward(retain_graph=True)
            optimizer.step()


        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, y0, t).to(device)
                
                pred_x = pred_y[:, 0]
                pred_v = pred_y[:, 1]
                pred_a = (-c*pred_v - k*pred_x)/m

                pred_E = 0.5*m*pred_v**2 + 0.5*k*pred_x**2   # total energy
                pred_P = -c*pred_v*pred_x                   # power

                mse = nn.MSELoss()(pred_y, y)
                violation = torch.sum(torch.relu(-E[0] + E))/len(t) + torch.sum(torch.abs(pred_P - pred_P[-1]))/len(t)

                print('Iter {:04d} | PHI_THETA {:.9f} | PHI_BEST {:.9f} | MSE {:.9f} | Penalty {:.9f} | Counter {:1d}'.format(itr, phi_theta.item(), phi_best.item(), mse.item(), violation.item(), counter))
     
                
                if itr == args.niters:
                    pred_y_test = odeint(func, y0, t_test).to(device)
                    
                    pred_x_test = pred_y_test[:, 0]
                    pred_v_test = pred_y_test[:, 1]
                    pred_a_test = (-c*pred_v_test - k*pred_x_test)/m

                    pred_E_test = 0.5*m*pred_v_test**2 + 0.5*k*pred_x_test**2   # total energy
                    pred_P_test = -c*pred_v_test*pred_x_test                   # power

                    mse_t = nn.MSELoss()(pred_y_test, test_y)
                    violation_t = torch.sum(torch.relu(-E[0] + E))/len(t_test) + torch.sum(torch.abs(pred_P_test - pred_P_test[-1]))/len(t_test)

                    print('MSE Test {:.6f} | Violation {:.6f}'.format(mse_t.item(), violation_t.item()))
                    
                    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
                    axs[0].plot(t_test.detach().cpu().numpy(), pred_x_test.detach().cpu().numpy(), color='green', label='predicted')
                    axs[0].set_ylabel("Displacement")
                    axs[1].plot(t_test.detach().cpu().numpy(), pred_v_test.detach().cpu().numpy(), color='blue', label='predicted')
                    axs[1].set_ylabel("Velocity")
                    axs[2].plot(t_test.detach().cpu().numpy(), pred_a_test.detach().cpu().numpy(), color='orange', label='predicted')
                    axs[2].set_ylabel("Acceleration")
                    axs[2].set_xlabel("Time")

                    axs[0].plot(t_test.detach().cpu().numpy(), x_test.detach().cpu().numpy(), linestyle='dashed', color='green', label='real')
                    axs[0].set_ylabel("Displacement")
                    axs[1].plot(t_test.detach().cpu().numpy(), v_test.detach().cpu().numpy(), linestyle='dashed', color='blue', label='real')
                    axs[1].set_ylabel("Velocity")
                    axs[2].plot(t_test.detach().cpu().numpy(), a_test.detach().cpu().numpy(), linestyle='dashed', color='orange', label='real')
                    axs[2].set_ylabel("Acceleration")
                    axs[2].set_xlabel("Time")
                    plt.legend()

                    plt.savefig(args.savePlot, format='eps')
                    torch.save(func, args.saveModel)

                    
