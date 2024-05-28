# BayOTIDE

(This repo is still on update)

This authors' official PyTorch implementation for paper:"**Bayesian Online Multivariate Time Series Imputation with Functional Decomposition**"[[OpenReview]](https://openreview.net/forum?id=aGBpiEcB8z&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2024%2FConference%2FAuthors%23your-submissions))[[Arxiv](https://arxiv.org/abs/2308.14906)] (ICML 2024).

---
#### Key Idea:  Decompose Multivariate Time as Latent Functions Factors + Online Filtering


<!-- <!-- <div align=center> <img src="./figs/FunBat-eq.PNG" width = 100%/> </div> -->

<div align=center> <img src="notebook/figs/fig_simu_impute.png" width = 100%/> </div>

---




<!-- Example of latent functions of spatial and temporal modes learned from real-world data.
<div align=center> <img src="./figs/FunBat.PNG" width = 100%/> </div>
<div align=center> <img src="./figs/FunBat-time.PNG" width = 50%/> </div> -->

## Requirements:
The project is mainly built with **pytorch 2.3.0** under **python 3.10**. The detailed package info can be found in `requirement.txt`.

## Instructions:
1. Clone this repository.
2. To play with the model quickly, we offer several notebooks at `notebook`(on synthetic & real data)
3. To run the real-world datasets with scripts, see `run_script.sh` for example.
4. To tune the (hyper)parametrs of model, modify the `.yaml` files in `config` folder
5. To apply the model on your own dataset, please follow the [process_script](https://github.com/xuangu-fang/BayOTIDE/tree/master/data/process_script) to process the raw data into appropriate format.
6. GPU choice: the models are run on CPU by default, but you can change the device to CPU by modifying the `device` as `cpu` of `.yaml` files in the `config` folder.


## Data

We offer the [raw data](https://drive.google.com/drive/folders/1DQJFZ9IkKw9pzr_vBSCLnrzqn4dp4kBd?usp=drive_link), [processed scripts](https://github.com/xuangu-fang/Functional-Bayesian-Tucker-Decomposition/tree/master/data/process_script), and processed data([Beijing](https://github.com/xuangu-fang/Functional-Bayesian-Tucker-Decomposition/tree/master/data/beijing), for all three datasets used in paper. The code for generating the synthetic data is also provided in the [data]( https://github.com/xuangu-fang/Functional-Bayesian-Tucker-Decomposition/tree/master/data/synthetic) folder.


If you wanna customize your own data to play the model, please follow the [process_script](https://github.com/xuangu-fang/BayOTIDE/tree/master/data/process_script).


## Citation

Please cite our work if you would like it
```
@misc{fang2023bayotide,
      title={BayOTIDE: Bayesian Online Multivariate Time series Imputation with functional decomposition}, 
      author={Shikai Fang and Qingsong Wen and Yingtao Luo and Shandian Zhe and Liang Sun},
      year={2023},
      eprint={2308.14906},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

