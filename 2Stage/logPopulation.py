import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

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
parser.add_argument('--nn', default='INODE')
parser.add_argument('--nrun', type=int, default=1)
parser.add_argument('--savePlot', type=str)
parser.add_argument('--saveModel', type=str)
parser.add_argument('--tf', type=int, default=300)
parser.add_argument('--tf_test', type=int, default=300)
parser.add_argument('--flag', type=int, default=0)
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
    start = time.perf_counter()

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.Adam(func.parameters(), lr=1e-5)
    end = time.time()

    pred_y = odeint(func, true_y0, t, method='rk4').to(device)
    pen = torch.mean((torch.maximum(torch.subtract(pred_y, 12), torch.Tensor([0]).to(device))))

    print("starting penalty phase training...")
    while pen >= 1e-4 or (ii <= 20 and pen <= 1e-4):
        ii += 1
        pen.backward()
        optimizer.step()
        pred_y = odeint(func, true_y0, t, method='rk4').to(device)
        pen = torch.mean((torch.maximum(torch.subtract(pred_y, 12), torch.Tensor([0]).to(device))))
        if ii % args.test_freq == 0:
            print('Iter {:04d} | Penalty {:.6f}'.format(ii, pen.item()))
        optimizer.zero_grad()

    

    best_f = 10000000
    best_params = list(func.parameters())
    best_pen = 1e-4
    pen_curr = pen
    loss_curr = 100000000000

    print("starting objective function training...")
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_y = odeint(func, true_y0, t, method='rk4').to(device)
        loss = nn.MSELoss()(pred_y, true_y)

        if pen_curr <= best_pen: 
            best_pen = pen_curr
            best_params = list(func.parameters())

        loss.backward()
        previous_params = list(func.parameters())
        optimizer.step()
        updated_params = list(func.parameters())

        pred_y_new = odeint(func, true_y0, t, method='rk4').to(device)
        pen_new = torch.mean((torch.maximum(torch.subtract(pred_y_new, 12), torch.Tensor([0]).to(device))))
        loss_new = nn.MSELoss()(pred_y_new, true_y)

        if pen_new > pen_curr:
            if args.flag == 1:
                for prev, upd in zip(previous_params, updated_params):
                    upd.data = prev.data
            elif args.flag == 2:
                pen_curr = best_pen
                for best, upd in zip(best_params, updated_params):
                    upd.data = best.data
            else:
                pen_curr = pen_new
                loss_curr = loss_new


        else:
            pen_curr = pen_new
            loss_curr = loss_new



        if itr % args.test_freq == 0 or itr == 1:
            with torch.no_grad():
                print('Iter {:04d} | MSE Loss {:.6f} | Penalty {:.6f}'.format(itr, loss_new.item(), pen_curr.item()))

                end = time.time()
        
                
                if itr == args.niters:
                    elapsed = (time.perf_counter() - start)
                    pred_y_test = odeint(func, true_y0, t_test)
                    mse_t = nn.MSELoss()(pred_y_test, test_y)
                    print('MSE Loss {:.6f}'.format(mse_t.item()))
                    plt.plot(t_test.detach().cpu().numpy(), test_y.detach().cpu().numpy(), linestyle='dashed', label='real')
                    plt.plot(t_test.detach().cpu().numpy(), pred_y_test.detach().cpu().numpy(), label='predicted')
                    plt.xlabel("Time")
                    plt.ylabel("Population")
                    plt.legend()
                    if args.test_data_size == args.data_size and args.tf_test == args.tf : 
                        a = "same_test_train"
                    elif args.test_data_size != args.data_size:
                        a = "different_number_points"
                    else:
                        a = "different_length"

                    if args.flag == 0: 
                        b = "simple" 
                    elif args.flag == 1: 
                        b = "updPrev" 
                    else: 
                        b = "updBest"
                    plt.savefig(args.savePlot)
                    torch.save(func, args.saveModel)
                                        
                    constraint = torch.maximum(torch.subtract(pred_y_test, 12), torch.Tensor([0]).to(device))
                    print("Constraint violated in: ", sum(list(map(lambda x: x>1e-6, constraint))).item(), " times!")
                    print("Total difference is ", torch.sum(constraint).item())
                    print("Total number of parameters updates skipped: ", skip_counter)
                    print("Best model at itr: ", best_itr, "with loss=", best_f)
                    print("Elapsed time is ", elapsed)
