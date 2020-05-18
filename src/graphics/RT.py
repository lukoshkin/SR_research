import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from .utils import minstrX


def displayRTresults(xdata, loss_logs, mode, explosion_ratio, w, i):
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9, 4))
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()

    xdata = xdata[:i+1]
    loss_logs = loss_logs.cpu().numpy()[:i+1].T
    mllog = loss_logs.mean(0)
    wllog = w.cpu().numpy() @ loss_logs
    x_min_str, x_min = minstrX(xdata, mllog)
    x_wmin_str, x_wmin = minstrX(xdata, wllog)

    viridis = cm.get_cmap('viridis', len(loss_logs))
    cmap = viridis(np.linspace(0, 1, len(loss_logs)))

    lbls = ['do', 'ic']
    repeats = len(loss_logs) - len(lbls)
    lbls += ['bc'] * repeats
    axes = [ax1, ax2] + [ax2] * repeats
    suffix = 'es' if repeats else ''

    for lbl, c, ax, loss in zip(lbls, cmap, axes, loss_logs):
        ax.plot(xdata, loss, c=c, alpha=.4, label=lbl)
    lines = ax1.lines + ax2.lines
    fig.suptitle('LR exponential range test')
    ax1.tick_params(axis='y', labelcolor=cmap[0])
    ax1.axvline(x_min, 0, 1, c='red', ls='--', alpha=.5)
    ax1.axvline(x_wmin, 0, 1, c='pink', ls='--', alpha=.5)
    ax2.set_ylabel(','.join(lbls[1:])+' loss'+suffix)
    ax1.set_ylabel('do loss')
    ax1.set_xlabel('iteration #')
    ax1.legend(lines, lbls, loc=9);
    if mode == 'expo':
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax3.set_xscale('log')

    ax3.plot(xdata, mllog, alpha=.6, c='dodgerblue')
    ax4.plot(xdata, wllog, alpha=.6, c='darkorange')
    lines = ax3.lines+ax4.lines
    lbls = ['arithmetic mean', 'weighted']
    ax3.legend(lines, lbls, loc=9)

    ax3.set_ylabel('mean loss')
    ax4.set_ylabel("loss terms' weighted sum")
    ax3.axvline(x_min, 0, 1, c='red', ls='--', alpha=.5)
    ax3.axvline(x_wmin, 0, 1, c='pink', ls='--', alpha=.5)
    ax3.text(
            .05, -.16, f'extremum at: LR = {x_min_str}',
            transform=ax3.transAxes, c='red', alpha=.5)
    ax3.text(
            .05, -.22, f'weighted loss extremum = {x_wmin_str}',
            transform=ax3.transAxes, c='pink')
    if explosion_ratio is not None:
        ax1.set_ylim(
                bottom=.9*loss_logs[0].min(),
                top=explosion_ratio*loss_logs[0, 0])
        ax2.set_ylim(
                bottom=.9*loss_logs[1:].min(),
                top=explosion_ratio*loss_logs[1:, 0].max())
        ax3.set_ylim(
                bottom=.9*mllog.min(),
                top=explosion_ratio*mllog[0])
        ax4.set_ylim(
                bottom=.9*wllog.min(),
                top=explosion_ratio*wllog[0])

    fig.tight_layout()
    fig.subplots_adjust(top=0.88);
    print('Minimum loss value (mean):', mllog.mean())
