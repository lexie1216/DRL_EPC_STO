# -*- ecoding: utf-8 -*-
# @ModuleName: main3.py
# @Function: 
#  
# @Time: 2024/5/8 12:42
import numpy as np
from ppo_discrete import PPO_discrete
import argparse
import logging
from meta_env import EpidemicModel
import pickle
import warnings

warnings.filterwarnings("ignore")


def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger


def evaluate_policy(args, env, agent, state_norm):
    num_episodes = 1000
    rewards = []
    overloads = []
    intensities = []
    sdos = []
    fdos = []
    tdos = []
    ados = []

    for _ in range(num_episodes):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False

        while not done:

            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, info = env.step(np.array(a))

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            s = s_

        ep_r, ep_overload, ep_intensity, ep_sdo, ep_fdo, ep_tdo, ep_ado = info.values()

        # env.render()
        rewards.append(ep_r)
        overloads.append(ep_overload)
        intensities.append(ep_intensity / env.period / env.ZONE_NUM)
        tdos.append(ep_tdo / env.ZONE_NUM)
        sdos.append(ep_sdo / env.period)
        fdos.append(ep_fdo / env.period)
        ados.append(ep_ado / env.period)

    result_dict = {
        'reward': rewards,
        'IOR': overloads,
        'ACI': intensities,
        'ATO': tdos,
        'ASO_adj': sdos,
        'ASO_mob': fdos,
        'ASO_adm': ados
    }

    return result_dict


def my_test(args):
    model_idx = args.model_idx
    eval_env = EpidemicModel(city=args.city, R0=args.R0)
    args.zone_num = eval_env.ZONE_NUM
    agent = PPO_discrete(args)
    agent.load(model_idx)

    for i in range(1):

        res_dict = evaluate_policy(args, eval_env, agent, args.use_state_norm)

        res[experiment_scenario[args.experiment_idx]] = res_dict

        np.save(f"res/model_{args.city}_{str(args.experiment_idx)}_{str(args.model_idx)}_{args.R0}_simRes.npy",
                np.array(eval_env.simRes))
        np.save(f"res/model_{args.city}_{str(args.experiment_idx)}_{str(args.model_idx)}_{args.R0}_actions.npy",
                np.array(eval_env.actions))

        eval_env.close()


parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
parser.add_argument("--max_train_steps", type=int, default=int(1.2e6), help=" Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=1.2e4,
                    help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--save_freq", type=int, default=10, help="Save frequency")
parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")

parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=3, help="PPO parameter")
parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.1, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

parser.add_argument("--state_dim", type=int, default=int(7), help="state dimension")
parser.add_argument("--action_dim", type=int, default=int(3), help="action dimension")

parser.add_argument("--hidden_width", type=int, default=256,
                    help="The number of neurons in hidden layers of the neural network")

parser.add_argument("--lr_a", type=float, default=1e-5, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate of critic")

parser.add_argument("--load_model", type=bool, default=False, help="load pretrained model or Not")
parser.add_argument("--model_idx", type=int, default=int(100), help="which model to load")
parser.add_argument("--experiment_idx", type=int, default=int(1),
                    help="experiment id determining the reward function and save path")

parser.add_argument("--city", type=str, default='nyc', help="case")
parser.add_argument("--R0", type=str, default='low', help="scenario")

args = parser.parse_args()

experiment_scenario = {
    4: 'basic',
    2: 't-order',
    3: 's-order-adj',
    5: 's-order-mob',
    -1: 's-order-adm',
    1: 'st-order-adj',
    6: 'st-order-mob',
    -2: 'st-order-adm'
}

if __name__ == '__main__':

    # args.city = 'sz'
    # args.R0 = 'high'
    # eids = [4, 2, 3, 5, -1, 1, 6, -2]
    # # mids = [80, 99, 98, 99, 100, 100, 100, 100]
    # mids = [80, 99, 98, 100, 100, 100, 100, 100]
    # for city in ['tokyo', 'nyc']:
    for city in ['tokyo']:
        args.city = city
        args.R0 = 'high'

        eids = [4, 3, 5]
        mids = [100, 100, 100]
        res = dict()

        for i in range(len(eids)):
            args.experiment_idx = eids[i]
            args.model_idx = mids[i]

            my_test(args)

        with open(f'res/evaluation_index_{args.city}_{args.R0}.pickle', 'wb') as f:
            pickle.dump(res, f)
