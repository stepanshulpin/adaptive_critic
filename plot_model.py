import torch
import torch.nn as nn
from data import SimODEData
from torchdiffeq import odeint as odeint
import pylab as plt
from typing import Tuple

def plot_time_series(true_model: torch.nn.Module, # true underlying model for the simulated data
                     fit_model: torch.nn.Module, # model fit to the data
                     data: SimODEData, # data set to plot (scatter)
                     time_range: tuple = (0.0, 30.0), # range of times to simulate the models for
                     ax: plt.Axes = None, 
                     dyn_var_idx: int = 0,
                     title: str = "Model fits",
                     *args,
                     **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the true model and fit model on the same axes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        
    ts = torch.linspace(time_range[0], time_range[1], 1000)
    ts_data, y_data = data

    initial_conditions = y_data[0, :].unsqueeze(0)
    sol_pred = odeint(fit_model, initial_conditions, ts, method='dopri5').detach().numpy()
    sol_true = odeint(true_model, initial_conditions, ts, method='dopri5').detach().numpy()
        
    ax.plot(ts, sol_pred[:,:,dyn_var_idx], color='skyblue', lw=2.0, label='Predicted', **kwargs);
    ax.scatter(ts_data.detach(), y_data[:,dyn_var_idx].detach(), color='black', s=30, label='Data',  **kwargs);
    ax.plot(ts, sol_true[:,:,dyn_var_idx], color='black', ls='--', lw=1.0, label='True model', **kwargs);
    ax.set_title(title);
    ax.set_xlabel("t");
    ax.set_ylabel("y");
    plt.legend();
    return fig, ax

def plot_phase_plane(true_model: torch.nn.Module, # true underlying model for the simulated data
                     fit_model: torch.nn.Module, # model fit to the data
                     data: SimODEData, # data set to plot (scatter)
                     time_range: tuple = (0.0, 30.0), # range of times to simulate the models for
                     ax: plt.Axes = None, 
                     dyn_var_idx: tuple = (0,1),
                     title: str = "Model fits",
                     *args,
                     **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the true model and fit model on the same axes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        
    ts = torch.linspace(time_range[0], time_range[1], 1000)
    ts_data, y_data = data
    
    initial_conditions = y_data[0, :].unsqueeze(0)
    sol_pred = odeint(fit_model, initial_conditions, ts, method='dopri5').detach().numpy()
    sol_true = odeint(true_model, initial_conditions, ts, method='dopri5').detach().numpy()
    
    idx1, idx2 = dyn_var_idx
    
    ax.plot(sol_pred[:,:,idx1], sol_pred[:,:,idx2], color='skyblue', lw=1.0, label='Fit model');
    ax.scatter(y_data[:,idx1], y_data[:,idx2].detach(), color='black', s=30, label='Data');
    ax.plot(sol_true[:,:,idx1], sol_true[:,:,idx2], color='black', ls='--', lw=1.0, label='True model');
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(title)
    return fig, ax