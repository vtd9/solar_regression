# Run in interactive session: exec(open('driver.py').read())

from re import X
import numpy as np
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch modules
import torch
from torch import nn
import torch.utils.data as Data


def load_array(data_arrays, batch_size, is_train=True): 
  '''Construct a PyTorch data iterator.'''
  dataset = Data.TensorDataset(*data_arrays)
  return Data.DataLoader(dataset, batch_size, shuffle=is_train)

def synthetic(w, b, num_examples, stdev=0.1):
  '''Generate y = Xw + b + noise.'''
  X = torch.normal(0, 1, (num_examples, len(w)))
  y = torch.matmul(X, w) + b
  y += torch.normal(0, stdev, y.shape)

  print('X.shape:', X.shape)
  return X, torch.reshape(y, (-1, 1))

def synthetic_only1(w, b, num_examples, stdev=0.1):
  '''y only depends on 1st input'''
  X = torch.normal(0, 1, (num_examples, len(w)))
  y = torch.matmul(X[:, 0].reshape(-1, 1), w[0].unsqueeze(0)) + b
  y += torch.normal(0, stdev, y.shape)

  print('X.shape:', X.shape)
  return X, torch.reshape(y, (-1, 1))

def synthetic_quad(w, b, num_examples, stdev=0.1):
  '''Generate y = X0**2 + noise. Other input has no effect...'''
  X = torch.normal(0, 1, (num_examples, len(w)))
  y = X[:, 0]**2 + b
  y += torch.normal(0, stdev, y.shape)
  return X, torch.reshape(y, (-1, 1)) 

def train(data_iter, net, loss_fun, opt, num_epochs=3, print_jacobian=False):
    # Train
    for epoch in range(num_epochs):
        for X, y in data_iter:
            # Forward
            X.requires_grad = True
            yhat = net(X)
            loss = loss_fun(yhat, y)

            # Backprop
            opt.zero_grad()
            loss.backward(retain_graph=True)
            if print_jacobian:
                dyhat_dX, = torch.autograd.grad(yhat, X, torch.ones(yhat.shape))
                print(dyhat_dX)
            opt.step()
        loss = loss_fun(net(X), y)
        print(f'epoch {epoch}, loss {loss:f}')
    return X, y

def test_linear(w, b, gen_fun, net, batch_size=10, lr=0.005, momentum=0.9):
    # Generate data
    features, labels = gen_fun(w, b, 1000)

    # Define optimizer
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # Train
    return train(load_array((features, labels), batch_size=batch_size), 
        net, nn.MSELoss(), opt, 
        num_epochs=5, print_jacobian=False)

def plot_fit(gen_fun, w, b, model, num_examples=1000, idx=0):
    x, y = gen_fun(w, b, num_examples)
    x = x.detach()
    y = y.detach()
    yhat = model(x).detach()
    plt.plot(x[:, idx], y, '.', label='synthetic data')
    plt.plot(x[:, idx], yhat, 'x', label='predicted')
    plt.legend()
    plt.show()

def test_a_net(gen_fun=synthetic, use_net1=False, plot=True, hand_calc=False):
    # Parameters in underlying formula
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    if use_net1:
        net = nn.Sequential(nn.Linear(2, 1))
    else:
        net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    X, y = test_linear(true_w, true_b, gen_fun, net, batch_size=5)
    X = X.detach()
    y = y.detach()

    if hand_calc:
        w1 = net[0].weight.detach()
        w2 = net[2].weight.detach()
        z1 = X @ w1.T + net[0].bias.detach()
        a1 = nn.ReLU()(z1)
        yhat = a1 @ w2.T + net[2].bias.detach()

    if plot:
        plot_fit(gen_fun, true_w, true_b, net)

   # Muck around with jacobian
    last_jacobian = torch.autograd.functional.jacobian(net, X)
    sum_over_samples = last_jacobian.sum(dim=0)[0]
    print('X:', X)
    print('Last jacobian, sum of dim 0:', last_jacobian.sum(dim=0))

    # True gradients
    if gen_fun == synthetic_only1:
        true_deriv = true_w.repeat(sum_over_samples.shape[0], 1)[:, 0]
        est_deriv = sum_over_samples[:, 0]
        # Sometimes can get pretty low mse difference, like 0.00113, 0.0832
    elif gen_fun == synthetic_quad:
        true_deriv = 2*X[:, 0]
        est_deriv = sum_over_samples[:, 0]
    else:
        true_deriv = true_w
        est_deriv = sum_over_samples
    print('\ncompare w true gradient:', true_deriv - est_deriv,
        '\nmse_diff:', nn.MSELoss()(true_deriv, est_deriv).item())
    
if __name__ == '__main__':
    test_a_net(synthetic_quad)