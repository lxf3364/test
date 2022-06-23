# ！-*- coding:utf-8 -*-
"""
@File  : Recommended_mol.py
@Author: Xiangfeng Li
@Date  : 2021/10/20 9:14
"""
import os

os.chdir('/'.join(__file__.split('/')[:-2]))

import requests
# from networks.LatentGanTransfer.latentgan_export import LatentGAN
import torch
import numpy as np
import torch.nn as nn
import ray
import math
import pandas as pd

from typing import List, Callable, Dict
from utils.basic_types import SMILES
import json
from utils.filter_smiles import RaySMILESFilter
from utils.workflow import check_cb
from utils.smiles_tsfm import RaySMILES2Graph
from agent.dqn.agent import BaselineDQNAgent
from rdkit.Chem import RDConfig, MolFromSmiles
from networks.predictor.molmap.molmap.example.mmp import MolMapProcess
import sys, os

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer
import pickle

from rdkit.Chem.Descriptors import qed

# from networks.rl.dqn import PerciveNet, QNet, DQN

N_CUPS = 5
N_GPUS = 2
N_CPUS_EACH = 2
N_GPUS_EACH = 1


# ray.init(num_cpus=N_CUPS, num_gpus=N_GPUS, ignore_reinit_error=True)


class LatentGanFPEncoder(nn.Module):
    def __init__(self, n_fp_bit: int, n_h: int, n_hiddens=(256, 256), p_h: float = 0.) -> None:
        super().__init__()
        layers = []
        layers.append(nn.Sequential(
            nn.Linear(n_fp_bit, n_hiddens[0]),
            nn.BatchNorm1d(n_hiddens[0]),
            nn.ReLU()
        ))

        for l, n_hidden in enumerate(n_hiddens):
            if l == 0:
                continue
            layers.append(nn.Sequential(
                nn.Linear(n_hiddens[l - 1], n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU()
            ))

        layers.append(nn.Sequential(
            nn.Linear(n_hiddens[-1], n_h),
            nn.Tanh()
        ))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


LatentGanFPEncoderConfig = dict(n_fp_bit=2048, n_h=256,
                                n_hiddens=(256, 256))


class OutputAction(nn.Module):
    """
    模拟一个强化学习网络的输出，当强化学习网络训练好时，则将此网络代替。
    """

    def __init__(self):
        super().__init__()

    def pick_action(self):
        return 75


rl_net = OutputAction()


# models = get_parallel_models()


class ActivateLearningDQNAgent(BaselineDQNAgent):
    """
    一个继承于BaselineDQNAgent类的通过主动学习挑选分子的类
    """
    def __init__(self, filter_workers: List[Callable], smiles2graph_workers: List[Callable],
                 generators: nn.Module, estimator, rl_net: nn.Module,
                 n_max_mols: int = 100,
                 n_sample_each: int = 1024, bs_sample: int = 1024, bs_estimate: int = 1024,
                 gen_train_kwargs: Dict = {}, est_train_kwargs: Dict = {}, N_WORKERS=None, init_data=None
                 ) -> None:
        """

        :param filter_workers: 用来过滤分子的一些方法
        :param smiles2graph_workers: 将smiles转化为图的工具
        :param generators: 分子生成器，在这个过程中，使用了request来代替
        :param estimator: 对于分子的活性预测器，主体为chemprop，其中集成了由request来提供的molmap。
        :param rl_net: 强化学习的网络
        :param n_max_mols: 挑选的分子的个数
        :param n_sample_each: 每轮生成的分子数量（但是都在分子生成器内部有另外的定义）
        :param bs_sample:
        :param bs_estimate:
        :param gen_train_kwargs:
        :param est_train_kwargs:
        :param N_WORKERS:
        :param init_data:
        """
        super().__init__(filter_workers, smiles2graph_workers, generators, estimator, rl_net,
                         n_max_mols=20, n_sample_each=n_sample_each, bs_sample=bs_sample,
                         bs_estimate=bs_estimate,
                         gen_train_kwargs=gen_train_kwargs, est_train_kwargs=est_train_kwargs
                         )

        self.init_data = init_data
        self.mol_pool = []
        self.models = estimator
        self.validated = []
        self.preds = []
        self.top_mean = n_max_mols
        self.init_weights()
        print('Agent Initialation Successful')

    def init_weights(self):
        ray.get([model.init_weights.remote() for model in self.models])

    def train_estimator(self, data):
        return ray.get([model.fine_tune.remote(data) for model in self.models])

    def estimate(self, mols):
        return ray.get([model.estimate.remote(mols) for model in self.models])

    def check_smiles(self, mols):
        """
        检查输入的分子是否有无效分子，
        :param mols: list[smiles]
        :return: list:无效的分子
        """
        return ray.get([model.check_smiles.remote(mols) for model in self.models])

    def get_molmap_prediction(self, train_data, pred_mols):
        """
        molmap的预测值
        :param train_data: 训练的数据
        :param pred_mols: 预测的数据，应为生成的分子
        :return:
        """
        url = 'http://172.16.200.124:1111/get_molmap_predict'
        data = {'train_data': train_data,
                'pred_data': pred_mols}
        pred = requests.post(url, json=data)
        return pred.json()

    def get_confidence(self, datas, target_mols, use_molmap):
        if not use_molmap:

            metrics = self.get_val_error(datas).mean()
        else:
            #print('****TRUE_MOLMAP')
            chemprop_metrics = self.get_val_error(datas)
            metrics = np.append(chemprop_metrics, self.get_molmap_var()).mean()
        loss = (1 / ((math.e) ** (metrics * 0.8)))
        simlarity = self.get_similar(datas, target_mols)
        new_simlarity = [1 / (math.e ** (-10 * (sim - 0.5)) + 1) for sim in simlarity]
        confidence = [round((float(smi) * float(loss)) ** 0.5, 2) for smi in new_simlarity]

        return confidence

    def get_train_MSE(self, datas):
        self.init_weights()
        self.train_estimator(datas)
        mols = [data[0] for data in datas]
        all_prediction_1 = np.array(self.estimate(mols))
        self.init_weights()
        self.train_estimator(datas)
        all_prediction_2 = np.array(self.estimate(mols))
        all_prediction = np.vstack((all_prediction_1, all_prediction_2)).mean(0).tolist()
        true_ic50 = [data[1] for data in datas]
        dif = [(true_ic50[i] - all_prediction[i]) ** 2 for i in range(len(datas))]
        mse = sum(dif) / len(datas)
        return mse

    def get_test_var(self, datas, mols):
        all_prediction = self.mol_mean_IC50(datas, mols)
        vars = all_prediction.var(0).tolist()
        return vars
        pass

    def get_molmap_var(self):
        import json
        with open('/home/drugfarm/WebService/MedChemistAI1/data/history.json',
                  'r') as f:
            stdout = json.load(f)
        var = stdout['val_rmse'][-1]
        os.remove('/home/drugfarm/WebService/MedChemistAI1/data/history.json')
        return var

    def get_val_error(self, datas):
        metrics = np.array([np.array(msg[1]).mean() for msg in self.train_estimator(datas)])
        #print('met', metrics)
        return metrics

    def set_device(self, device: torch.device):
        nets = [self.rl]
        for net in nets:
            try:
                net.set_device(device)
                net.device = device
            except:
                net.to(device)

    @check_cb
    def select_action(self):
        return self.rl.pick_action()

    @check_cb
    def step(self, data, username, use_molmap, rate):
        self.data_pool += data
        action = self.select_action()
        # self.train_estimator(self.data_pool, **self.est_train_kwargs)
        selected_mols ,selected_act= self.interpret_action(data, action, username, use_molmap, rate)
        print('sel_mol', selected_mols)
        self.n_step += 1
        return selected_mols,selected_act

    @check_cb
    def train_generators(self, data, *args, **kwargs):
        pass

    @check_cb
    def sample_mols(self, mols, username):
        print('生成分子...')
        url = 'http://172.16.200.124:6664/get_mols'  # latentgan_server
        # url = 'http://172.16.200.124:1234/get_mmpdb_mol'  # mmpdb_sever
        # url = 'http://192.168.31.215:8864/get_mols'  #mmolMPT_server
        data = {'mols': mols,
                'id': username}
        #print(data)
        req = requests.post(url, json=data)
        # print(req)
        return [req.json()]

    @check_cb
    def interpret_action(self, data, action, username, use_molmap, rate):
        mols_for_gen_train = self.pick_mols_for_generator_training()
        # print(mols_for_gen_train)
        # print(username)

        # sample new mols
        #sample_mols = self.sample_mols(mols_for_gen_train, username)
        print('分子生成结束，开始过滤。')
        sample_mols = pd.read_csv("/home/drugfarm/WebService/MedChemistAI1/server/DF3C021_round3_mollib(1).csv")['SMILES'].tolist()
        sample_mols = [sample_mols]
        print(sample_mols)
        print(len(sample_mols[0]))
        to_kick = [o[0] for o in self.data_pool]
        #print(type(sample_mols[0]))
        filtered_mols = [self.filter_mols(mols) for mols in sample_mols]
        print('filtered_mols', len(filtered_mols[0]))
        #print('第一次过滤...')
        filtered_mols = [self.kick_out_checked(
            mols, to_kick) for mols in filtered_mols]
        # simlar = self.get_similar(data, filtered_mols[0])
        print('filtered_mols', len(filtered_mols[0]))

        if use_molmap:
            print('使用molmap')
            assert 0 < rate < 1, '同时使用molmap与chemprop时，他们之间存在一种加权求和的关系。'
            mmp_pred = self.get_molmap_prediction(data, filtered_mols[0])
            # mmp_pred = [i for l in mmp_pred for i in l]
            #print(mmp_pred)
            print('mmp预测结束...')
            chemprop_pred = self.mol_mean_IC50(data, filtered_mols[0])
            chemprop_pred_mean = chemprop_pred.mean(0).tolist()
            means = [((1 - rate) * mmp_pred[i] + rate * chemprop_pred_mean[i]) for i in range(len(filtered_mols[0]))]
            stds = np.vstack((mmp_pred, chemprop_pred)).std(0).tolist()
        else:
            all_prediction = self.mol_mean_IC50(data, filtered_mols[0])
            means = all_prediction.mean(0).tolist()
            stds = all_prediction.std(0).tolist()
        score_smi = self.calculate_score(data, filtered_mols[0], means)

        # score_smi = [(score[i] * simlar[i]) ** 0.5 for i in range(len(filtered_mols[0]))]

        all_data = {'smiles': filtered_mols[0],
                    'means': means,
                    'score_smi': score_smi,
                    'stds': stds}

        dataframe = pd.DataFrame(all_data).sort_values(by='means', ascending=True)
        # selection_bymeans = dataframe[:int(action)].copy()
        # dataframe_sort_stds = dataframe[int(action):int(len(dataframe) * 0.5)].sort_values(by='stds', ascending=False)
        # selection_bystd = dataframe_sort_stds[:self.top_mean - int(action)]
        # dataframe_concat = pd.concat([selection_bymeans, selection_bystd])
        # print('dataframe_concat', dataframe_concat)
        # selection_mol = dataframe_concat.smiles.tolist()
        # selection_act = dataframe_concat.means.tolist()
        selection_mol = dataframe.smiles.tolist()
        selection_act = dataframe.means.tolist()

        return selection_mol, selection_act

    def store_weights(self):
        pass

    def get_similar(self, datas, target_mols):
        """
        :param target_mols: 需要被计算相似度的分子
        :param datas: 给定训练的分子
        :return:target_mols对datas的相似性，长度为len(target_mols)
        """
        from rdkit import Chem
        from rdkit import DataStructs
        # print(len((datas)))
        smiles1 = [data[0] for data in datas]
        smiles1 = smiles1[:20]
        # print(smiles1)
        fps1 = []
        for smi in smiles1:
            m = Chem.MolFromSmiles(smi)
            if m != None:
                fps1.append(Chem.RDKFingerprint(m))
        fps2 = []
        for smi in target_mols:
            m = Chem.MolFromSmiles(smi)
            if m != None:
                fps2.append(Chem.RDKFingerprint(m))
        similarity = []
        for fp2 in fps2:
            sm = []
            for fp1 in fps1:
                sm.append(DataStructs.FingerprintSimilarity(fp1, fp2))
            # sm = np.max(np.array(sm))
            similarity.append(max(sm))
        return similarity

        # pass

    def mol_mean_IC50(self, data, mols):
        """
        for item in data:
            if not isinstance(item[1], float):
                data.remove(item)
        train_smiles = [o[0] for o in data]
        invalid_smiles = self.check_smiles(train_smiles)
        if len(invalid_smiles) == 0:
            data = data
        else:
            data = [item for item in data if item[0] not in invalid_smiles]
        """
        self.init_weights()
        self.train_estimator(data)
        all_prediction_1 = np.array(self.estimate(mols))
        self.init_weights()
        self.train_estimator(data)
        all_prediction_2 = np.array(self.estimate(mols))
        all_prediction = np.vstack((all_prediction_1, all_prediction_2))
        return all_prediction



    def calculate_score(self, data, smiles, activities, max_activity: float = 5):
        score_list = []
        # print(type(activities[0]))

        act_score = [(max_activity - float(activity)) / max_activity + 0.1 for activity in activities]
        # print('act_score:', act_score)
        # act_score=[a_d/max_activity + 0.1 for a_d in activity_data]
        for i, smile in enumerate(smiles):
            mol = MolFromSmiles(smile)
            qed_score = qed(mol)
            try:
                sa = sascorer.calculateScore(mol)
            except ZeroDivisionError:
                sa = 100
            sa_score = (1 / sa / 0.9 - 1 / 9)
            # print('sa',sa_score,)
            score = (act_score[i] * qed_score * sa_score) ** (1 / 6)
            # score = (activities[i] * qed_score * sa_score) ** (1 / 6)
            # print('acti',activities[i])
            # print('***score', score)
            # score = round(score, 2)
            # 
            score_list.append(score)
        simlar = self.get_similar(data, smiles)
        score_smi = [(score_list[i] * simlar[i]) ** 0.5 for i in range(len(smiles))]
        return score_smi

    def qed_sa(self, smiles, max_activity: float = 5):
        score_list = []
        sa_list=[]
        qed_list=[]
        # print(type(activities[0]))

        # act_score = [(max_activity - float(activity)) / max_activity + 0.1 for activity in activities]
        # print('act_score:', act_score)
        # act_score=[a_d/max_activity + 0.1 for a_d in activity_data]
        for i, smile in enumerate(smiles):
            mol = MolFromSmiles(smile)
            qed_score = qed(mol)
            qed_list.append(qed_score)
            try:
                sa = sascorer.calculateScore(mol)
                sa_list.append(sa)
            except ZeroDivisionError:
                sa = 100
            sa_score = (1 / sa / 0.9 - 1 / 9)
            score = (qed_score * sa_score) ** 0.5
            score = round(score, 2)
            # print('***score', score)
            score_list.append(score)

        return qed_list,sa_list

    def restore(self) -> None:
        self.data_pool = []
        self.obs_pool = []
        self.processed_obs_pool = []
        self.actions = []

        self.n_step = 0
        self.y_min = 10 ** 10

    def process_obs(self):
        last_obs = self.obs_pool[-1]
        ys = [o[1] for o in last_obs]
        new_min = min(ys)
        if new_min < self.y_min:
            self.y_min = new_min
        estimates = [o[2] for o in last_obs]

        errors = [abs(np.mean(estimates[i]) - ys[i]) for i in range(len(ys))]
        error_mean = sum(errors) / len(errors)

        proc_obs = [self.y_min, error_mean]
        self.processed_obs_pool.append(proc_obs)
        return proc_obs

    # def get_MCTS(self, MCTS):
    #     """
    #     :param MCTS:一个MCTS对象，其中包括分子及其活性，Q与N的值等属性。
    #     :return:
    #     """
    #     self.MCTS_mol = MCTS.mol
    #     self.MCTS_act = MCTS.act
    #     self.MCTS_Q = MCTS.Q
    #     self.MCTS_N = MCTS.N
    #     pass
    #
    # def get_MCTS_score(self,):



def make_agent(models):
    print('Agent Building')
    agent_kwargs = {'n_sample_each': 2000}

    filter_smiles_workers = [
        RaySMILESFilter.remote() for i in range(N_CPUS_EACH)
    ]

    smiles2grpah_workers = [
        RaySMILES2Graph.remote() for i in range(N_CPUS_EACH)
    ]

    estimator = models

    # char_rnn = get_char_rnn_model()

    device = torch.device('cuda:2')

    # print('device')
    agent = ActivateLearningDQNAgent(filter_smiles_workers, smiles2grpah_workers,
                                     [None], estimator, rl_net,
                                     **agent_kwargs)
    agent.set_device(device)
    return agent
