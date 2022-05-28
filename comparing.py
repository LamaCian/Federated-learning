
#comparing optimizers
optimizers = ["fedavg", "fedprox"]
datsets = ["Leukemia", "Covid", "Breast Cancer"]

rounds = []
accuracy = []
loss = []
loss_fedavg_leukemia = []
for dataset in datasets:
    for optimizer in optimizers:
        #tff.learning.USE_OPTIMIZER
        #calculate loss, accuracy, rounds
        # if optimizer == 'fedavg':
            # if dataset == 'Leukemia':
                # loss_fedavg.append(loss) or # fed_avg[loss] = loss ; fed_avg[acc] = accuracy
            # elif dataset == 'Covid':
            #     ...
        

def compare():
    """ Compares loss and accuracy for one dataset using 2 different optimizers """
    #plot loss of fedavg, fedprox through num_rounds 
    #
    ...


# create table of loss, metrics..
#save weights to specific folder 
#output_dir --> save weights there
#for testing load everything from output_dir
# save start of training ->
#output_dir -> 





        



    
