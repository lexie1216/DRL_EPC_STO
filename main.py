import torch
import numpy as np
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from datetime import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
import os, shutil
import logging
from meta_env import EpidemicModel

import warnings
import antropy as ant

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
    times = 1
    evaluate_reward = 0
    evaluate_sorder = 0
    evaluate_torder = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        episode_sorder = 0
        episode_torder = 0
        step = 0
        while not done:
            step = step + 1

            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, info = env.step(np.array(a))
            s_order = info['s_order']
            t_order = info['t_order']
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            episode_sorder += s_order
            episode_torder += t_order
            # print(step,info['day'],s_order)
            # if step % 5 == 1 and step>10 and step<35:
            #     print(info['day'],info['s_order'])

            s = s_
        env.render()
        evaluate_reward += episode_reward
        evaluate_sorder += episode_sorder
        evaluate_torder += episode_torder
        max_I = np.max(np.sum(env.simRes[:, :, 2], axis=1), axis=0)
        sum_level = np.sum(env.actions)

        evaluate_t_ent = 0
        for i in range(env.actions.shape[1]):
            a = env.actions[:, i]
            p_e = ant.perm_entropy(a,order=10)
            evaluate_t_ent += p_e

        evaluate_torder = np.sum(abs(env.actions[1:] - env.actions[:-1]), axis=0).sum()

    return round(evaluate_reward / times, 2), round(max_I, 0), sum_level, round(evaluate_sorder / times, 2), round(
        evaluate_torder / times, 2), evaluate_t_ent


def my_test(args):
    model_idx = args.model_idx
    eval_env = EpidemicModel()
    agent = PPO_discrete(args)
    agent.load_with_id(model_idx)
    sum_reward = 0
    for i in range(1):
        e_reward, e_max_I, e_sum_level, e_s_order, e_t_order ,e_t_ent= evaluate_policy(args, eval_env, agent,
                                                                               args.use_state_norm)
        sum_reward += e_reward
        overload = (e_max_I - eval_env.capacity) / eval_env.capacity
        overload = round(overload * 100, 2)
        print(
            'experiment:{}\t model:{}\t ep_reward:{}\t max_infections:{}({}%)\t sum_level:{}\t spatial_entropy:{} temporal_change:{}'.format(
                args.experiment_idx, args.model_idx, e_reward, e_max_I, overload, e_sum_level, e_s_order, e_t_order))

        print(
            'experiment:{}\t model:{}\t overload_ratio:{}%\t average_intensity:{}\t spatial_entropy:{} temporal_change:{} perm_ent:{}'.format(
                args.experiment_idx, args.model_idx, overload, round(e_sum_level / 120 / 74, 4),
                round(e_s_order / 120, 4), round(e_t_order / 74, 4),round(e_t_ent / 74, 4)))

        np.save('model%d/simRes%d.npy' % (args.experiment_idx, model_idx), np.array(eval_env.simRes))
        np.save('model%d/actions%d.npy' % (args.experiment_idx, model_idx), np.array(eval_env.actions))
        eval_env.close()


def main(args, seed):
    env = EpidemicModel(args.experiment_idx)
    env_evaluate = EpidemicModel(args.experiment_idx)
    # Set random seed
    env.seed(seed)
    env_evaluate.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.max_episode_steps = env.period  # Maximum number of steps per episode
    args.zone_num = env.ZONE_NUM  # Maximum number of steps per episode

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)
    # 使用预训练模型
    if args.load_model:
        agent.load_with_id(args.model_idx)
        evaluate_num += args.model_idx

    # Build a tensorboard

    timenow = str(datetime.now())[0:-10]
    timenow = '_' + timenow[0:10] + '_' + timenow[-5:-3] + '-' + timenow[-2:] + '_'
    #     timenow = '_2023-06-04_12-10'
    writepath = 'runs/epidemic' + timenow + str(args.experiment_idx)
    if os.path.exists(writepath): shutil.rmtree(writepath)
    writer = SummaryWriter(log_dir=writepath)

    #######LOGGING#######
    log_path = os.path.join("logs", "epidemic" + timenow + str(args.experiment_idx) + ".log")
    if not os.path.exists(f"logs"):
        os.makedirs(f"logs")

    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(args)
    logger.info(
        f'task starts: {str(datetime.now())[0:-10]} ,state_dim: {args.state_dim} action_dim: {args.action_dim} zone_num: {args.zone_num}')

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
                evaluate_reward, e_max_I, e_sum_level, e_s_order, e_t_order,e_t_ent = evaluate_policy(args, env_evaluate, agent,
                                                                                              state_norm)
                evaluate_rewards.append(evaluate_reward)

                writer.add_scalar('eval_Index/ep_r', evaluate_reward, global_step=total_steps)
                writer.add_scalar('eval_Index/max_I', e_max_I, global_step=total_steps)
                writer.add_scalar('eval_Index/sum_Level', e_sum_level, global_step=total_steps)
                writer.add_scalar('eval_Index/spatial_order', e_s_order, global_step=total_steps)
                writer.add_scalar('eval_Index/temporal_change', e_t_order, global_step=total_steps)
                writer.add_scalar('eval_Index/temporal_entropy', e_t_ent, global_step=total_steps)
                logger.info(
                    f'evaluate_num:{evaluate_num} starts: {str(datetime.now())[0:-10]} \t evaluate_reward:{evaluate_reward}\t max_I:{e_max_I} \t sum_level:{e_sum_level}\t spatial_entropy:{e_s_order}\t temporal_change:{e_t_order}\t temporal_ent:{e_t_ent}\t')

                agent.save(evaluate_num)


parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
parser.add_argument("--max_train_steps", type=int, default=int(6e6), help=" Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=1.2e4,
                    help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--save_freq", type=int, default=10, help="Save frequency")
parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
parser.add_argument("--hidden_width", type=int, default=256,
                    help="The number of neurons in hidden layers of the neural network")
parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
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
parser.add_argument("--zone_num", type=int, default=int(74), help="zone num")

parser.add_argument("--load_model", type=bool, default=False, help="load pretrained model or Not")
parser.add_argument("--model_idx", type=int, default=int(200), help="which model to load")
parser.add_argument("--experiment_idx", type=int, default=int(1),
                    help="experiment id determining the reward function and save path")

args = parser.parse_args()

if __name__ == '__main__':
    # args.experiment_idx = 4
    # args.model_idx = 100
    # args.load_model = True
    # args.lr_a = 1e-6
    # args.lr_c = 1e-6
    # args.max_train_steps = int(1.2e6)
    #
    # main(args,seed=3047)

    args.model_idx = 200

    # for i in range(4):
    #     args.experiment_idx = i + 1
    #     my_test(args)
    for i in [2,3]:
        args.experiment_idx = i + 1
        my_test(args)

    # args.experiment_idx = 4
    # args.model_idx = 300
    # my_test(args)
    # args.experiment_idx = 4
    # args.model_idx = 301
    # my_test(args)
