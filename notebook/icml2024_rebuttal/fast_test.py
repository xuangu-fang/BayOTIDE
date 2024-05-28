import numpy as np
import torch 
import sys
sys.path.append("../../")

import tqdm
import yaml
torch.random.manual_seed(300)
import matplotlib.pyplot as plt

import utils_BayTIDE as utils

from model_BayTIDE import BayTIDE

from model_LDS import LDS_GP_streaming

import os

import fire


def evals(dataset = "guangzhou", D_trend =3, D_season=10):
        
    # config_path = "./config_{%s}.yaml"
    config_path = "./config_%s.yaml" % dataset


    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        

    data_file = config["data_path"] # T=315 - rmse = 0.27 - Mqar23

    hyper_dict = utils.make_hyper_dict(config)

    hyper_dict["n_season"] =  D_season
    hyper_dict["K_trend"] = D_trend


    data_dict = utils.make_data_dict(hyper_dict,data_file,fold=0)

    model = BayTIDE(hyper_dict,data_dict)

    model.reset()
    model.post_W_m = torch.ones_like(model.post_W_m).to(model.device)

    running_rmse = []
    # _, loss_dict = model.model_test(0)
    # running_rmse.append(loss_dict["test_RMSE"])


    INNER_ITER = hyper_dict["INNER_ITER"]
    EVALU_T = hyper_dict["EVALU_T"]
    for epoch in range(1):
        model.reset()
        for T_id in tqdm.tqdm(range(model.T)):
            model.filter_predict(T_id)
            model.msg_llk_init()

            # for inner_it in range(INNER_ITER):
            for inner_it in range(INNER_ITER):
            


                flag = (inner_it == (INNER_ITER - 1))

                model.msg_approx_W(T_id)
                model.post_update_W(T_id)

                model.msg_approx_U(T_id)
                model.filter_update(T_id,flag)

                model.msg_approx_tau(T_id)
                model.post_update_tau(T_id)

            if  T_id == model.T - 1:

            #     # we only need  this to get the running metric, otherwise we can skip this
            #     # model.inner_smooth()
                
                _, loss_dict = model.model_test(T_id,True)
            #     print("T_id = {}, train_rmse = {:.2f}, test_rmse= {:.2f},test_MAE= {:.2f}".format(T_id, loss_dict["train_RMSE"],
            #      loss_dict["test_RMSE"], loss_dict["test_MAE"]))
            #     running_rmse.append(loss_dict["test_RMSE"])

        # _, loss_dict = model.model_test(T_id, True)
        before_smooth_loss = "before smooth, test_rmse = {:.2f}, test_MAE= {:.2f}, test_CRPS={:.3f}".format(loss_dict["test_RMSE"], loss_dict["test_MAE"],loss_dict["CRPS"])
        
        print(before_smooth_loss)
        
        
        
        model.smooth()
        model.post_update_U_after_smooth(0)
        _, loss_dict = model.model_test(T_id, True)
        
        after_smooth_loss = "after smooth, test_rmse = {:.2f}, test_MAE= {:.2f}, test_CRPS={:.3f}".format(loss_dict["test_RMSE"], loss_dict["test_MAE"],loss_dict["CRPS"])
        
        print(after_smooth_loss)
        
        # save the results as txt file
        dic_name = "./result_log/%s/D_season-%d/"%(dataset, hyper_dict["n_season"])
        
        file_name =  dic_name + "D_trend-%d.txt"%(hyper_dict["K_trend"])
        
        # make sure the file esixts otherwise create it
        
        from pathlib import Path
        
        Path(  dic_name).mkdir(parents=True, exist_ok=True)
        
        with open(file_name, "a") as f:
            f.write(before_smooth_loss + "\n")
            f.write(after_smooth_loss + "\n")
            f.write("\n")
            f.close()
        

    
if __name__ == '__main__':
    
    fire.Fire(evals) 
