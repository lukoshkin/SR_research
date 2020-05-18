# Experiments

Notes on training the equation `du/dt + a du/dx = ru(1-u)`

1. RNNLike; new tools; 100 epochs, bs=128, loss ~ MAE -- loss100/best: .042/.040
1. RNNLike; new tools; 320 epochs with better lr scheduler strategy, bs=128, loss ~ MAE -- loss320/best: .013/.012
1. RNNLike; new tools; 320 epochs with better lr scheduler strategy, bs=128, loss ~ MAE, increased batch gen domain (4x) -- loss320/best: .002/.0015  
                                                                                for MSE score is different, cannot compare them directly, but performance is  worse
1. RNNLike; new tools, with RAR(2); 100 epochs, bs=128, loss ~ MAE -- loss100/best: .065/.064


The one of successfull settings:
lr: starting from 1e-1
    100 epochs with gamma .9 every 2 steps
    220 epochs with gamma .979 every step
T: [0, 4]


Offtop: loss balancing starting ratio: 35/2.5 - try to apply it to other experiments

# Conclusions

***These are the points that lead to a better network learning:***

1. Balance among loss terms
1. Good lr scheduler strategy (in networks and optimizers that account for doing it manually)
1. Expanded sampling domain
