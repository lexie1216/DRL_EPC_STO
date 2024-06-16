import torch
import numpy as np
from utils.normalization import Normalization, RewardScaling
from utils.replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from datetime import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
import os, shutil
import logging
from meta_env import EpidemicModel

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
    num_episodes = 1
    total_rewards, total_overload, total_intensity, total_sdo, total_fdo, total_tdo, total_ado = 0, 0, 0, 0, 0, 0, 0

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

        env.render()

        total_rewards += ep_r
        total_overload += ep_overload
        total_intensity += ep_intensity
        total_sdo += ep_sdo
        total_fdo += ep_fdo
        total_tdo += ep_tdo
        total_ado += ep_ado

    average_reward = total_rewards / num_episodes
    average_overload = total_overload / num_episodes
    average_intensity = total_intensity / num_episodes
    average_sdo = total_sdo / num_episodes
    average_fdo = total_fdo / num_episodes
    average_tdo = total_tdo / num_episodes
    average_ado = total_ado / num_episodes

    return (
        average_reward,
        average_overload,
        average_intensity,
        average_sdo,
        average_fdo,
        average_tdo,
        average_ado
    )


def my_test(args):
    model_idx = args.model_idx
    eval_env = EpidemicModel(city=args.city, R0=args.R0)
    args.zone_num = eval_env.ZONE_NUM
    agent = PPO_discrete(args)
    agent.load(model_idx)

    for i in range(1):
        e_r, e_overload, e_intensity, e_sdo, e_fdo, e_tdo, e_ado = evaluate_policy(args, eval_env, agent,
                                                                                   args.use_state_norm)
        print(
            'experiment:%d\t model:%d\t reward:%.2f\t overload_ratio:%.2f%%\t intensity:%.3f\t sdo:%.3f tdo:%.3f fdo:%.3f ado:%.3f' %
            (args.experiment_idx, args.model_idx,
             e_r,
             e_overload * 100,
             e_intensity / eval_env.period / eval_env.ZONE_NUM,
             e_sdo / eval_env.period,
             e_tdo / eval_env.ZONE_NUM,
             e_fdo / eval_env.period,
             e_ado / eval_env.period)
        )

        np.save(f"res/model_{args.city}_{str(args.experiment_idx)}_{str(args.model_idx)}_{args.R0}_simRes.npy",
                np.array(eval_env.simRes))
        np.save(f"res/model_{args.city}_{str(args.experiment_idx)}_{str(args.model_idx)}_{args.R0}_actions.npy",
                np.array(eval_env.actions))
        eval_env.close()


def main(args, seed):
    env = EpidemicModel(reward_mode=args.experiment_idx, city=args.city, R0=args.R0, use_trick=args.use_trick)
    env_evaluate = EpidemicModel(reward_mode=args.experiment_idx, city=args.city, R0=args.R0, use_trick=args.use_trick)
    # Set random seed
    env.seed(seed)
    env_evaluate.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.max_episode_steps = env.period  # Maximum number of steps per episode
    args.zone_num = env.ZONE_NUM

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)
    # 使用预训练模型
    if args.load_model:
        agent.load(args.model_idx)
        evaluate_num += args.model_idx

    # Build a tensorboard

    timenow = str(datetime.now())[0:-10]
    timenow = '_' + timenow[0:10] + '_' + timenow[-5:-3] + '-' + timenow[-2:] + '_'
    writepath = 'runs/' + args.city + timenow + str(args.experiment_idx) + "_" + args.R0
    if os.path.exists(writepath): shutil.rmtree(writepath)
    writer = SummaryWriter(log_dir=writepath)

    # LOGGING
    log_path = os.path.join("logs", args.city + timenow + str(args.experiment_idx) + "_" + args.R0 + ".log")
    if not os.path.exists(f"logs"):
        os.makedirs(f"logs")

    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(args)
    logger.info(
        f'task starts: {str(datetime.now())[0:-10]}')

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob, _ = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(np.array(a))

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                a_loss, c_loss, entropy, lr_now = agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

                writer.add_scalar('train_loss/a_loss', a_loss, global_step=total_steps)
                writer.add_scalar('train_loss/c_loss', c_loss, global_step=total_steps)
                writer.add_scalar('train_loss/entropy', entropy, global_step=total_steps)
                writer.add_scalar('train_loss/lr', lr_now, global_step=total_steps)

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1

                e_r, e_overload, e_intensity, e_sdo, e_fdo, e_tdo, e_ado = evaluate_policy(args,
                                                                                           env_evaluate,
                                                                                           agent,
                                                                                           state_norm)
                evaluate_rewards.append(e_r)

                writer.add_scalar('eval_Index/ep_r', e_r, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_overload', e_overload, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_intensity', e_intensity, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_sdo', e_sdo, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_tdo', e_tdo, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_fdo', e_fdo, global_step=total_steps)
                writer.add_scalar('eval_Index/ep_ado', e_ado, global_step=total_steps)

                logger.info(
                    'evaluate_num:%d starts: %s\t reward:%.2f\t overload:%.2f%% \t intensity:%.3f\t sdo:%.3f\t tdo:%.3f\t fdo:%.3f\t ado:%.3f\t' %
                    (evaluate_num, str(datetime.now())[0:-10], e_r, e_overload * 100,
                     e_intensity / env.period / env.ZONE_NUM,
                     e_sdo / env.period,
                     e_tdo / env.ZONE_NUM,
                     e_fdo / env.period,
                     e_ado / env.period)
                )

                agent.save(evaluate_num)

    for handler in logger.handlers:
        handler.close()
    logger.handlers = []
    logging.shutdown()


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

if __name__ == '__main__':
    # training
    for city in ['sz','nyc','tokyo']:
        args.city = city
        for level in ['high']:
            args.R0 = level
            for mode in [4, 2, 3, 5, -1, 1, 6, -2]:
                args.experiment_idx = mode

                args.lr_a = 3e-4
                args.lr_c = 3e-4
                args.load_model = False

                main(args, seed=3047)


