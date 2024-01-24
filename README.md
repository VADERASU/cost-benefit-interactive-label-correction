
# A Simulation Based Approach for Quantifying Human Benefits in Interactive Label Correction (Supplementary Materials)


- Simulation Code
- Simulation Result Raw Data
- Simulation Result Report Data
- Statistical Model



## Reproduce the Simulation

### Simulation Code
We included the following example python scripts for both FashionMNIST and AGNews-10pct simulation in their individual folders. The scripts for these two datasets are slight different but the methodologies and functions are the same.
| Code File                                     | Description                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------------- | 
| noise_generate.py          |      Generating three types of noise, should be imported to `dataset_noise_generation.ipynb`|
| dataset_noise_generation.ipynb      | Sampling and corrupting data. | 
| simulation_random_single.py         | Simulating human-assisted label correction. should be imported to `simulation.ipynb`    | 
| simulation.ipynb   | Triggering the simulation.| 


**Note** Running `dataset_noise_generation.ipynb` firstly for sampling and generating corrupted data, you will get three more data folders (logits_and_preds, sampled_data, noisy_data), which are used in running `simulation.ipynb` for simulating human relabeling corrupted dataset. We did random sampling during the simulation. Thus, the generated data and result might be different if you reproduce the simulation. 
**Note** These scripts only illustrate few examples, you can adjust variable values for changing the condition setup based on the paper.
**Note** Before running these python scripts, changing the directories for saving and reading data in the code.



### Simulation Result Raw Data 
Including four simulation result data files for FashionMNIST binary classification, FashionMNIST multi-class classification, AGNews-10ct binary classification, and AGNews-10pct multi-class classification. Theses data are used for statistical testing and analysis.


### Simulation Result Report Data 
Including the aggregate data that reported in the manuscript Section 5, including average values and associated standard deviation values.


### Statistic Model
We built generalized linear models (JMP$\circledR$ version 16) using full factorial combination of five simulation factors (excluding Dataset Complexity $|\mathcal{L}|$), with two evaluation metrics ($R(\beta, 0)$ and $D(\beta, 0)$) as response variables. These constructed generalized linear models are stored in PDF files.  

| Dataset | Response Variable | Task | Filename |
| -----------------------------------------|-------------------------------------------|-------------------------------------------| -------------------------------------------| 
|    FashionMNIST     |  D(β,0)   |  Binary Classification  | FashionMNIST_D(β,0)_binary.pdf|
|     FashionMNIST     |  D(β,0)   |  Multi-class Classification  | FashionMNIST_D(β,0)_multi.pdf|
|     FashionMNIST     |  R(β,0)   |  Binary Classification  | FashionMNIST_R(β,0)_binary.pdf|
|     FashionMNIST     |  R(β,0)   |  Multi-class Classification  |FashionMNIST_R(β,0)_multi.pdf |
|    AGNews-10pct     |  D(β,0)   |  Binary Classification  | AGNews-10pct_D(β,0)_binary.pdf|
|     AGNews-10pct     |  D(β,0)   |  Multi-class Classification  | AGNews-10pct_D(β,0)_multi.pdf|
|     AGNews-10pct     |  R(β,0)   |  Binary Classification  | AGNews-10pct_R(β,0)_binary.pdf|
|     AGNews-10pct     |  R(β,0)   |  Multi-class Classification  | AGNews-10pct_R(β,0)_multi.pdf|


### Real-world Noise Validation Code
We included the python scripts for examining the effect of human-assisted label correction and validating the simulation results on CIFAR-10N datasets (CIFAR-10 Worst, CIFAR-1O Aggregate, CIFAR-10 Random1, CIFAR-10 Random2, CIFAR-10 Random3).

 
 ### Real-world Noise Validation Result Raw Data
Including four validation result data files for CIFAR-10N datasets (CIFAR-10 Worst, CIFAR-1O Aggregate, CIFAR-10 Random1, CIFAR-10 Random2, CIFAR-10 Random3).