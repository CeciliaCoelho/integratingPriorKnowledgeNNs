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
parser.add_argument('--data_size', type=int, default=200)
parser.add_argument('--test_data_size', type=int, default=200)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--tf', type=int, default=300)
parser.add_argument('--tf_test', type=int, default=300)
parser.add_argument('--savePlot', type=str)
parser.add_argument('--saveModel', type=str)
parser.add_argument('--updateStrategy', type=str, choices=["True", "False"], default="True")
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([2.518629]).to(device)

t = torch.linspace(0., args.tf, args.data_size).to(device)
t_test = torch.linspace(0., args.tf_test, args.test_data_size).to(device)



class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mul(torch.mul(0.026, y), torch.sub(1, torch.div(y,12)))


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    test_y = odeint(Lambda(), true_y0, t_test, method='dopri5')


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50,50),
            nn.ELU(),
            nn.Linear(50, 1),
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

    ii = 0
    tol = 1e-4

    func = ODEFunc().to(device)
    optimizer = optim.Adam(func.parameters(), lr=1e-5)

    theta_best = list(func.parameters())
    theta = list(func.parameters())

    for itr in range(1, args.niters + 1):
        pred_y = odeint(func, true_y0, t, method='rk4').to(device)

#####################################################
        MSE_theta = torch.square(torch.subtract(pred_y, true_y))

        P_theta = 0 
        F_theta = torch.mean(normalizeFunc(MSE_theta))

        #inequality constraint
        v_j = torch.relu(torch.subtract(pred_y, 12))

        counter = sum(list(map(lambda x: x==0, torch.subtract(pred_y, 12)))).item() #to count how many constraints violations were =0

        if itr % 100 == 0 or itr == 1:
            mu_j = torch.div(torch.sum(v_j != 0), len(v_j))
        P_theta = torch.mean(v_j)

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
            phi_theta = torch.add(F_theta, torch.mul(mu_j, torch.divide(torch.sum(normalizeFunc(v_j)), len(v_j))))


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

        if itr != args.niters:
            optimizer.zero_grad()
            phi_theta.backward(retain_graph=True)
            optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                mse = nn.MSELoss()(pred_y, true_y)
                violation = torch.mean((torch.relu(torch.subtract(pred_y, 12))))
                print('Iter {:04d} | PHI_THETA {:.9f} | PHI_BEST {:.9f} | MSE {:.9f} | Penalty {:.9f} | Counter {:1d}'.format(itr, phi_theta.item(), phi_best.item(), mse.item(), violation.item(), counter))
                ii += 1

                end = time.time()
        
                
        if itr == args.niters:
            pred_y_test = odeint(func, true_y0, t_test)
            mse_t = nn.MSELoss()(pred_y_test, test_y)
            violation_t = torch.mean((torch.relu(torch.subtract(pred_y_test, 12))))
            print('MSE Test {:.9f} | Violation {:.9f}'.format(mse_t.item(), violation_t.item()))
            plt.plot(t_test.detach().cpu().numpy(), test_y.detach().cpu().numpy(), linestyle='dashed', label='real')
            plt.plot(t_test.detach().cpu().numpy(), pred_y_test.detach().cpu().numpy(), label='predicted')
            plt.xlabel("Time")
            plt.ylabel("Population")
            plt.legend()
            plt.savefig(args.savePlot, format='eps')
            torch.save(func, args.saveModel)
