''' 
    [!] TRAINING PARAMETERS [!]
'''


check = 5;
epochs = 100;
lr_init = 0.01;
batch_size = 16;
T_0 = 17;
T_mult = 2;
eta_min = 0.00001;

sched_params = dict();
sched_params["T_0"] = T_0;
sched_params["T_mult"] = T_mult;
sched_params["eta_min"] = eta_min;




