import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim




parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100)
parser.add_argument('--test_data_size', type=int, default=100)
parser.add_argument('--niters', type=int, default=4000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--nrun', default=1)
parser.add_argument('--savePlot', type=str)
parser.add_argument('--saveModel', type=str)
parser.add_argument('--tf', type=int, default=100)
parser.add_argument('--tf_test', type=int, default=100)
parser.add_argument('--flag', type=int, default=0)
args = parser.parse_args()


if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

start_time = 0
end_time = args.tf
end_time_test = args.tf_test
time_steps = args.data_size
time_steps_test = args.test_data_size
dt = (end_time - start_time) / time_steps
dt_test = (end_time_test - start_time) / time_steps_test

# Define the initial masses of each element in the reaction
A = 1
B = 1
C = 0
D = 0

# Define the reaction rate constants
k1 = 0.1
k2 = 0.05

# Store the masses of each element in time
times = np.arange(start_time, end_time, dt)
times_test = np.arange(start_time, end_time_test, dt_test)
masses = np.zeros((time_steps, 4))
masses_test = np.zeros((time_steps_test, 4))
masses[0,:] = [A, B, C, D]
masses_test[0,:] = [A, B, C, D]

# Simulate the reaction over time
for i, t in enumerate(times[:-1]):
    a = masses[i, 0]
    b = masses[i, 1]
    c = masses[i, 2]
    d = masses[i, 3]
    masses[i+1,:] = [a + dt * (-k1 * a * b),
                     b + dt * (-k1 * a * b + k2 * c),
                     c + dt * (k1 * a * b - k2 * c),
                     d + dt * (k1 * a * b)]

for i, t in enumerate(times_test[:-1]):
    a = masses_test[i, 0]
    b = masses_test[i, 1]
    c = masses_test[i, 2]
    d = masses_test[i, 3]
    masses_test[i+1,:] = [a + dt_test * (-k1 * a * b),
                     b + dt_test * (-k1 * a * b + k2 * c),
                     c + dt_test * (k1 * a * b - k2 * c),
                     d + dt_test * (k1 * a * b)]




# Normalize the masses so that the total mass is 1 at each time step
masses = masses / np.sum(masses, axis=1, keepdims=True)
masses_test = masses_test / np.sum(masses_test, axis=1, keepdims=True)





class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50,64),
            nn.ELU(inplace=True),
            nn.Linear(64,50),
            nn.Tanh(),
            nn.Linear(50, 4),
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

    true_y0 = torch.Tensor(masses[0,:]).to(device)
    true_y = torch.Tensor(masses).to(device)
    true_y_test = torch.Tensor(masses_test).to(device)
    t = torch.Tensor(times).to(device)
    t_test = torch.Tensor(times_test).to(device)
    tot_m = torch.sum(true_y0)
    mse_p = torch.Tensor([1])
    pen_p = torch.Tensor([0])


    pred_y = odeint(func, true_y0, t, method='rk4').to(device)
    pen = torch.mean(torch.abs(torch.subtract(torch.sum(pred_y, dim=1), tot_m)))
    print("starting penalty phase training...")
    while pen >= 1e-2 or (ii <= 20 and pen <= 1e-2):
        ii += 1
        pen.backward()
        optimizer.step()
        pred_y = odeint(func, true_y0, t, method='rk4').to(device)
        pen = torch.mean(torch.abs(torch.subtract(torch.sum(pred_y, dim=1), tot_m)))
        if ii % args.test_freq == 0:
            print('Iter {:04d} | Penalty {:.6f}'.format(ii, pen.item()))
        optimizer.zero_grad()


    best_f = 10000000
    best_params = list(func.parameters())
    best_pen = 1e-2
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
        pen_new = torch.mean(torch.abs(torch.subtract(torch.sum(pred_y_new, dim=1), tot_m)))
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

                
                if itr == args.niters:
                    elapsed = (time.perf_counter() - start)
                    pred_y_test = odeint(func, true_y0, t_test)
                    mse = nn.MSELoss()(pred_y_test, true_y_test)
                    print('MSE Test {:.6f}'.format(mse.item()))
                    plt.plot(times_test, masses_test[:, 0], label="A", linestyle='dashed', color='green')
                    plt.plot(times_test, masses_test[:, 1], label="B", linestyle='dashed', color='blue')
                    plt.plot(times_test, masses_test[:, 2], label="C", linestyle='dashed', color='orange')
                    plt.plot(times_test, masses_test[:, 3], label="D", linestyle='dashed', color='red')
                    plt.plot(times_test, pred_y_test[:, 0].detach().cpu().numpy(), label="A", color='green')
                    plt.plot(times_test, pred_y_test[:, 1].detach().cpu().numpy(), label="B", color='blue')
                    plt.plot(times_test, pred_y_test[:, 2].detach().cpu().numpy(), label="C", color='orange')
                    plt.plot(times_test, pred_y_test[:, 3].detach().cpu().numpy(), label="D", color='red')
                    plt.xlabel("Time (s)")
                    plt.ylabel("Mass")
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
                                        
                    total_mass = torch.sum(pred_y_test, dim=1)
                    conservation = [abs(total_mass[i]-total_mass[0]) for i in range(1,len(total_mass))]
                    print("Conservation of mass violated ", sum(list(map(lambda x: x>1e-6, conservation))).item(), " times!")
                    print("Total difference is ", sum(conservation).item())
                    print("Total number of parameters updates skipped: ", skip_counter)
                    print("Best model at itr: ", best_itr, "with loss=", best_f)
                    print("Elapsed time is ", elapsed)


