## Source code to reproduce experiments from "Spike-timing-dependent plasticity with activation-dependent scaling for receptive fields development"
This repository contains python source code of all experiments described in the paper  "Spike-timing-dependent plasticity with activation-dependent scaling for receptive fields development".
###  Setup:
+ To simulate spiking neural network (SNN) Brain2 dependencies need to be installed. 
    https://brian2.readthedocs.io/en/stable/introduction/install.html
+ Create 'data' directory in the project root folder, copy MNIST data set from http://yann.lecun.com/exdb/mnist/ 
to the directory and extract it.
+ Path to the data set can be customized by changing  DATA_PATH variable in **utils.py** script.
+ Project tree after extracting the data set:

```
.
├── classification.py
├── data
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
├── inhibition_scaling_experiment.py
├── rapid_scaling.py
├── SSNNetwork.py
└── utils.py

```


### Experiments:
##### Influence of competitiveness on RFs development:
To replicate results from "Influence of competitiveness on RFs development" section **inhibition_scaling_experiment.py** script needs to be executed. 
Script takes two command-line arguments inhibitory conductance *w_inh* and scaling time constant *taur*.
```console
python inhibition_scaling_experiment.py 20 0.1
``` 
##### Method evaluation in larger networks
To replicate results from "Method evaluation in larger networks"  section **classification.py** and **rapid_scaling.py** 
scripts need to be executed. 
To perform classification without rapid scaling **classification.py** script is executed. The script takes two command-line
arguments network size N, scaling constant Ro. 

```console
python classification.py 1000 0.2
``` 
Scripts generate output folder containing networks from each learning epoch. Scripts also check if training batch (6000_1batch_) and test batch (test_batch) are in the data folder if not - scripts generate them. Please notice that generating batches can take some time, and be aware that both batches occupy around ~ 1GB of the local storage.
To perform the rapid scaling, the **rapid_scaling.py** script needs to be executed. Scripts takes four arguments *Ro, *ext*, *start_batch_name*, *start_batch_folder*.
*Ro* argument describes higher scaling constant than in the classification phase, *ext* is the increase in excitatory time constant to perform classification after rapid scaling,
*start_batch_name* is the name of the network on which the rapid scaling is performed, and *start_batch_folder* is the folder generated during the classification phase, inside
*start_batch_folder*, *start_batch_name* network is kept. 

```console
python rapid_scaling.py 0.4 0.3 network0 classification_0.1_20
```
Scripts generates three adtional batches during classification '_eval' '_final_results','_iden', containg classification results 
and other data gathered during evalutaion of the network. For details please refer to the source code.
  

