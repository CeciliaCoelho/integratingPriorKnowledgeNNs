import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from filterMethodNN import *

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



if __name__ == '__main__':

    ii = 0
    filterT = []

    func = ODEFunc().to(device)

    bestF = 100000000; bestV = 100000000 
    best_params = list(func.parameters())
    bestIter = 0
    tolF = 1e-4
    
    optimizer1 = optim.Adam(func.parameters(), lr=1e-5)
    optimizer2 = optim.Adam(func.parameters(), lr=1e-5)
    end = time.time()

    for itr in range(1, args.niters + 1):
        pred_y = odeint(func, true_y0, t, method='rk4').to(device)
        loss = nn.MSELoss()(pred_y, true_y)
        violation = torch.mean((torch.maximum(torch.subtract(pred_y, 12), torch.Tensor([0]).to(device))))

        previous_params = list(func.parameters())

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss.backward(retain_graph=True)
        optimizer1.step()
        trialF_params = list(func.parameters())
        trialF_pred_y = odeint(func, true_y0, t, method='rk4').to(device)
        trialF_loss = nn.MSELoss()(trialF_pred_y, true_y)
        trialF_violation = torch.mean((torch.maximum(torch.subtract(trialF_pred_y, 12), torch.Tensor([0]).to(device))))

        current_params = trialF_params

        for prev, upd in zip(previous_params, current_params):
            upd.data = prev.data


        violation.backward(retain_graph=True)
        optimizer2.step()
        trialV_params = list(func.parameters())
        trialV_pred_y = odeint(func, true_y0, t, method='rk4').to(device)
        trialV_loss = nn.MSELoss()(trialV_pred_y, true_y)
        trialV_violation = torch.mean((torch.maximum(torch.subtract(trialV_pred_y, 12), torch.Tensor([0]).to(device))))

        current_params = trialV_params

        #verify trialV and trialF are feasible, keep best
        if trialF_violation <= tolF and trialV_violation <= tolF:
            if trialF_loss <= trialV_loss:
                updated_params = trialF_params
            else:
                updated_params = trialV_params
        elif trialF_violation <= tolF:
                updated_params = trialF_params
        elif trialV_violation <= tolF:
                updated_params = trialV_params
        else:
            filterT, acceptF = filterMethod(filterT, trialF_params, trialF_loss, trialF_violation)
            filterT, acceptV = filterMethod(filterT, trialV_params, trialV_loss, trialV_violation)

            if acceptV and acceptF:
                if trialF_violation <= trialV_violation:
                    updated_params = trialF_params
                else:
                    updated_params = trialV_params
            elif acceptV:
                    updated_params = trialV_params
            elif acceptF:
                    updated_params = trialF_params
            else:
                updated_params = previous_params


        for prev, upd in zip(updated_params, current_params):
            upd.data = prev.data



        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t, method='rk4')
                mse = nn.MSELoss()(pred_y, true_y)
                print('Iter {:04d} | MSE Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

                end = time.time()
        
                
                if itr == args.niters:
                    pred_y_test = odeint(func, true_y0, t_test)
                    mse_t = nn.MSELoss()(pred_y_test, test_y)
                    violation_t = torch.mean((torch.maximum(torch.subtract(pred_y_test, 12), torch.Tensor([0]).to(device))))
                    print('MSE Loss {:.6f} | Violation {:.6f}'.format(mse_t.item(), violation_t.item()))
                    plt.plot(t_test.detach().cpu().numpy(), test_y.detach().cpu().numpy(), linestyle='dashed', label='real')
                    plt.plot(t_test.detach().cpu().numpy(), pred_y_test.detach().cpu().numpy(), label='predicted')
                    plt.xlabel("Time")
                    plt.ylabel("Population")
                    plt.legend()
                    plt.savefig(args.savePlot)
                    torch.save(func, args.saveModel)
