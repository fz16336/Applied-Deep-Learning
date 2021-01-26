# Our code

The full code is provided and is fully commented for ease of understanding. When running the code, the following file arguments are available as adjustable hyperparameters:
- --*epochs* (int), 
- --*batchsize* (int)
- --*dropout* (float),
- --*optimiser* (str), 
- --*momentum* (float), 
- --*weight-decay* (float), 
- --*learning-rate* (float), 
- --*mode* (str) 
- --TSCNN. 

The optimisers available are 
- *Stochastic Gradient Decent* ('SGD')
- *Adaptive Learning Rate Optimisation* ('Adam') or
- *Adam with Decoupled Weight Decay Regularisation* (’AdamW’). 

Note, the type of network used, either
*LMCNet* ('LMC'), *MCNet* ('MC') or *MLMCNet* ('MLMC'), is
determined using --mode. However using --TSCNN overwrites
this for *TSCNNNet* and loads previously saved LMCNet and
MCNet results automatically.

To change the hyperparameters simply specify the arguments and its value as you call main.py on the command line. Like so:

    python main.py --epoch 10000 --learning-rate 0.01



