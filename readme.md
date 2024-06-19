# Code Instructions

The data and codes that used in this research are available on GitHub with the unique identifier at the link [https://github.com/lexie1216/DRL\_EPC\_STO](https://github.com/lexie1216/DRL\_EPC\_STO). 

## Introduction

A deep-reinforcement-learning-based (DRL) model to optimize epidemic control strategies with a focus on considering the coordination of interventions in both spatial and temporal dimensions.

## Usages

The values of "experiment_idx" and their corresponding experiments are as follows:

- `4` corresponds to "basic"
- `2` corresponds to "t-order"
- `3`, `5`, `-1` correspond to "s-order-adj", "s-order-mob", "s-order-adm", respectively
- `1`, `6`, `-2` correspond to "st-order-adj", "st-order-mob", "st-order-adm", respectively

### Training the Model

1. **Data Storage:**
   - Required data are stored in the `data/city` directory.
   - The files `adj_dict.pkl`, `flow_top3_dict.pkl`, and `adm_dict.pkl` contain the spatial interaction relationships for the three types of spatial orderliness: geographical adjacency of sub-regions, population mobility relationships, and administrative management relationships, respectively.
   - `flow.npy` is the intra-urban human mobility matrix.
   - `population.npy` contains the population distribution information.

2. **Training Setup:**
   - In `train.py`, you can set parameters such as the city, R0, ST-Order scenario, and DRL algorithm-related parameters.
   - Run `train.py` to start training.
   - The trained model checkpoints will be saved in the `model` directory.

### Testing the Trained Model

1. **Model Testing:**
   - `test.py` will test each model 1000 times (with the initial distribution of infected individuals being random each time). The results of these tests will be saved in the `res` directory as `evluation_index_{city_name}_{R_0}.pickle` files.
   - `evaluation_ttest.py` reads the above pickle files and performs a t-test to obtain the results for the table in the manuscript, which are saved in `evluation_results_{city_name}_{R_0}.xlsx` files.

2. **Control Strategy Visualization:**
   - To display the spatial-temporal distribution of control strategies in the manuscript, we fix the initial distribution of infected individuals for a single test and save the simulation results in `res/simRes.npy` and `actions.npy`.
   - The `utils` directory contains several scripts for plotting figures:
     - `plot_seir.py` can be used to plot the curves shown in Figure 5 and Figure 6c.
     - `plot_t_contrast.py` and `plot_s_contrast.py` are used to plot the comparisons of temporal orderliness and spatial orderliness in Figure 6a and Figure 6b, respectively.

[//]: # (### Example Commands)

[//]: #
[//]: # (#### Training)

[//]: #
[//]: # (```bash)

[//]: # (python train.py --city city_name --R0 R0_value --experiment_idx experiment_index --other_parameters ...)
