import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_smac import MAPPO_SMAC
from smac.env import StarCraft2Env
import networkx as nx
import matplotlib.pyplot as plt


from replay_buffer import ReplayBuffer

import datetime
class Runner_MAPPO_SMAC1:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.degree_array = []

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create environment
        self.env = StarCraft2Env(map_name=self.env_name, seed=self.seed, difficulty="6")
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = MAPPO_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Load pre-trained model if exists
        if True:
            print("Checking if a pre-trained model exists...")
            model_path = self.args.model_path
            if os.path.exists(model_path):
                print(f"Loading pre-trained model from {model_path}...")
                try:
                    #self.actor.load_state_dict(torch.load("./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
                    m='1000'
                    self.agent_n.load_model(self.args.model_path)
                    print(f"Model loaded successfully from {model_path}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
            else:
                print(f"Model path {model_path} does not exist.")
        else:
            print("No pre-trained model specified for loading.")

        # Further setup...

        # Create tensorboard for logging
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.win_rates = []  # Record the win rates
        self.total_steps = 0

        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self):
        evaluate_num = -1
        win_rates, evaluate_rewards = [], []

        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                win_rate, evaluate_reward = self.evaluate_policy()
                evaluate_num += 1
                win_rates.append(win_rate)
                evaluate_rewards.append(evaluate_reward)
                self.plt(win_rates, evaluate_rewards) 

            _, _, episode_steps = self.run_episode_smac(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()


        self.evaluate_policy()
        self.plt(win_rates, evaluate_rewards)
        
        model_name_prefix = f"{self.env_name}_seed_{self.seed}"
        self.agent_n.save_model(model_name_prefix, self.number, self.seed, self.total_steps)
        self.env.close() #


    def evaluate_policy(self):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
        print(f"total_steps:{self.total_steps} \t win_rate:{win_rate} \t evaluate_reward:{evaluate_reward}")
        return win_rate, evaluate_reward

    # Paint curve
    def plt(self, win_rates, rewards):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(win_rates)), win_rates, 'r-', label='win_rates')
        plt.xlabel('Evaluation Cycle')
        plt.ylabel('Win Rates')
        plt.title(f'Win Rate (Env: {self.env_name}, Seed: {self.seed})')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(range(len(rewards)), rewards, label='rewards')
        plt.xlabel('Evaluation Cycle')
        plt.ylabel('Episode Rewards')
        plt.title(f'Reward (Env: {self.env_name}, Seed: {self.seed})')
        plt.grid(True)
        
        plt.tight_layout()

        # 使用 os.path.join 保证路径在不同操作系统上的兼容性
        save_path_dir = os.path.join('D:', 'tie-smac_IMPLUSE', 'result', 'ppo', 'graph-1c3s5z')
        
        # 确保目录存在
        os.makedirs(save_path_dir, exist_ok=True)

        # 构建包含环境名和种子的唯一文件名
        base_filename = f'env_{self.env_name}_seed_{self.seed}'
        
        # 保存图片
        img_save_path = os.path.join(save_path_dir, f'{base_filename}_plot.png')
        plt.savefig(img_save_path, format='png')
        
        # 保存 win_rates 数据
        win_rates_save_path = os.path.join(save_path_dir, f'{base_filename}_win_rates.npy')
        np.save(win_rates_save_path, np.array(win_rates))

        # 保存 rewards 数据
        rewards_save_path = os.path.join(save_path_dir, f'{base_filename}_rewards.npy')
        np.save(rewards_save_path, np.array(rewards))

        print(f"Results for seed {self.seed} saved to {save_path_dir}")
        plt.close()

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
            self.agent_n.actor.msg_rnn_hidden = None
            self.agent_n.critic.msg_rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)

            # s = self.env.get_state()  # s.shape=(state_dim,)

            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)

            # # #get g graph node position
            pos_array = self.env.get_pos_array()

            dis_array = [[], [], [], [], [], [], [], [], []]

            # # #create g graph
            G = nx.Graph()

            # add node and edge
            for x in range(self.args.N):
                if pos_array[x][2] != 0:
                    G.add_node(x, pos=(pos_array[x][0], pos_array[x][1]))

                    for y in range(self.args.N):
                        if pos_array[y][2] != 0:
                            disx = pos_array[x][0] - pos_array[y][0]
                            disy = pos_array[x][1] - pos_array[y][1]

                            dis = np.sqrt(np.sum(np.square([disx, disy])))
                            if y == x:
                                dis_array[x].append(10000)
                            else:
                                dis_array[x].append(dis)
                        else:
                            dis_array[x].append(10000)  # node is dead or node is self
                else:
                    for j in range(self.args.N):
                        dis_array[x].append(10000)

            for index in range(self.args.N):
                for i in range(self.args.N):
                    if i != index and dis_array[index][i] <= pos_array[index][2] / 3 and pos_array[index][2] != 0 and \
                            pos_array[i][2] != 0:
                        G.add_edge(index, i, weight=dis_array[index][i])

            for index in range(self.args.N):
                if pos_array[index][2] != 0:
                    if G.degree(index) == 0:
                        min_index = dis_array[index].index(min(dis_array[index]))
                        G.add_edge(index, min_index, weight=min(dis_array[index]))
            # generate edge between sub graph， set init distance  1500
            tem_near = 1500
            temp_array = []
            node_i = 0
            node_j = 0
            # print(6)
            sub_nod_array = []
            for sub_g in nx.connected_components(G):
                sub_g = G.subgraph(sub_g)
                g_node = sub_g.nodes()
                sub_nod_array.append(g_node)

            # print(7)
            if len(sub_nod_array) > 1:
                big_sub = max(sub_nod_array, key=len)
                index = sub_nod_array.index(big_sub)
                sub_nod_array[index] = sub_nod_array[0]
                sub_nod_array[0] = big_sub

                # connect all the sub_graph
                i = len(sub_nod_array)
                for j in range(i):
                    for k in range(i):
                        if k > j:
                            for l in sub_nod_array[j]:
                                for m in sub_nod_array[k]:
                                    if dis_array[l][m] < tem_near:
                                        tem_near = dis_array[l][m]
                                        node_i = l
                                        node_j = m
                            G.add_edge(node_i, node_j, weight=tem_near)
                            tem_near = 150
                            node_i = 0
                            node_j = 0

            degree_array = [[0] * self.args.N, [0] * self.args.N, [0] * self.args.N, [0] * self.args.N,
                            [0] * self.args.N, [0] * self.args.N, [0] * self.args.N, [0] * self.args.N,
                            [0] * self.args.N
                            ]

            tie_strenth = [[0] * self.args.N, [0] * self.args.N, [0] * self.args.N, [0] * self.args.N,
                           [0] * self.args.N, [0] * self.args.N, [0] * self.args.N, [0] * self.args.N, [0] * self.args.N
                           ]

            # cpmpute tie strength distribution
            for i in range(self.args.N):
                for j in range(self.args.N):
                    if j > i:
                        if pos_array[i][2] > 0 and pos_array[j][2] > 0:
                            g_list = list(nx.shortest_path(G, i, j))
                            i_degree = G.degree(i)
                            j_degree = G.degree(j)
                            ij_strenth = (len(g_list)) / (i_degree + j_degree + len(g_list) - 2)
                            tie_strenth[i][j] = ij_strenth
                    else:
                        tie_strenth[i][j] = tie_strenth[j][i]

            temp = []
            for i in range(self.args.N):
                for j in range(self.args.N):
                    if tie_strenth[i][j] != 0:
                        temp.append(tie_strenth[i][j])
            if len(temp) == 0:
                threshold_strenth = 10
            else:
                temp.sort()

                threshold_strenth = temp[-int(len(temp) * 0.3)]

            # print(tie_strenth)
            # print(threshold_strenth)
            for i in range(self.args.N):
                for j in range(self.args.N):
                    if j > i:
                        if pos_array[i][2] > 0 and pos_array[j][2] > 0:
                            # print(str(tie_strenth[i][j])+'__'+str(threshold_strenth))
                            if tie_strenth[i][j] < threshold_strenth:
                                degree_array[i][j] = 1

                    else:
                        degree_array[i][j] = degree_array[j][i]

                if pos_array[i][2] != 0:
                    degree_array[i][i] = 1
            temp = []
            # print(9)
            for i in range(self.args.N):
                if pos_array[i][2] != 0:
                    temp.append(G.degree(i))
                else:
                    temp.append(0)
            max_degree_index = temp.index(max(temp))
            for i in range(self.args.N):
                if pos_array[i][2] != 0:
                    degree_array[i][max_degree_index] = 1
            self.degree_array = degree_array

            G.clear()

            own_feats = self.env.get_obs_own_feats_size()
            move_feats = self.env.get_obs_move_feats_size()
            n_enemies, n_enemy_feats = self.env.get_obs_enemy_feats_size()
            n_allies, n_ally_feats = self.env.get_obs_ally_feats_size()
            enemy_feats = n_enemies * n_enemy_feats
            ally_feats = n_allies * n_ally_feats

            # #select and cocat share obs
            # mmm
            # move_feats: 4
            # enemy_feats: (10, 8)
            # ally_feats: (9, 8)
            # own_feats: 4

            # 8m
            # move_feats: 4
            # enemy_feats: (8, 5)
            # ally_feats: (7, 5)
            # own_feats: 1

            # 2s3z
            # move_feats: 4
            # enemy_feats: (5, 8)
            # ally_feats: (4, 8)
            # own_feats: 4

            # 3s5z
            # move_feats: 4
            # enemy_feats: (8, 8)
            # ally_feats: (7, 8)
            # own_feats: 4

            temp = copy.deepcopy(obs_n)
            # for i in range(self.args.N):
            #     temp[i] = [0] * self.args.obs_dim
            for i in range(self.args.N):

                index = np.nonzero(np.array(self.degree_array[i]))
                # select Enemy features
                # mmm ind = 10
                for enemy_id in range(self.args.N):
                    if np.sum(obs_n[i][
                              own_feats + enemy_id * n_enemy_feats:own_feats + n_enemy_feats * (enemy_id + 1)] == 0):
                        for j in index[0]:
                            obs_j = self.env.get_other_obs_agent(j, i)
                            if np.sum(obs_n[j][own_feats + enemy_id * n_enemy_feats:own_feats + n_enemy_feats * (
                                    enemy_id + 1)] != 0):
                                temp[i][own_feats + enemy_id * n_enemy_feats:own_feats + n_enemy_feats * (
                                            enemy_id + 1)] = obs_j[
                                                             own_feats + enemy_id * n_enemy_feats:own_feats + n_enemy_feats * (
                                                                     enemy_id + 1)]
                                break

                # select ally features
                # mmm ind = 8
                for ally_id in range(self.args.N):
                    if ally_id < i:
                        if obs_n[i][own_feats + n_enemies * n_enemy_feats + ally_id * n_ally_feats] == 0:
                            for j in index[0]:
                                obs_j = self.env.get_other_obs_agent(j, i)
                                if j < ally_id and obs_n[j][
                                    own_feats + n_enemies * n_enemy_feats + (ally_id - 1) * n_ally_feats] != 0:
                                    temp[i][
                                    own_feats + n_enemies * n_enemy_feats + ally_id * n_ally_feats:own_feats + n_enemies * n_enemy_feats + n_ally_feats * (
                                                ally_id + 1)] = obs_j[
                                                                own_feats + n_enemies * n_enemy_feats + (
                                                                        ally_id - 1) * n_ally_feats:own_feats + n_enemies * n_enemy_feats + n_ally_feats * ally_id]
                                    break
                                elif j > ally_id and obs_n[j][
                                    own_feats + n_enemies * n_enemy_feats + (ally_id - 1) * n_ally_feats] != 0:

                                    temp[i][
                                    own_feats + n_enemies * n_enemy_feats + ally_id * n_ally_feats:own_feats + n_enemies * n_enemy_feats + n_ally_feats * (
                                                ally_id + 1)] = \
                                        obs_j[own_feats + n_enemies * n_enemy_feats + (
                                            ally_id) * n_ally_feats:own_feats + n_enemies * n_enemy_feats + n_ally_feats * (
                                                    ally_id + 1)]
                                    break
                    elif ally_id > i:
                        if obs_n[i][own_feats + n_enemies * n_enemy_feats + (ally_id - 1) * n_ally_feats] == 0:
                            for j in index[0]:
                                obs_j = self.env.get_other_obs_agent(j, i)
                                if j < ally_id and obs_n[j][
                                    own_feats + n_enemies * n_enemy_feats + (ally_id - 1) * n_ally_feats] != 0:
                                    temp[i][
                                    own_feats + n_enemies * n_enemy_feats + (
                                                ally_id - 1) * n_ally_feats:own_feats + n_enemies * n_enemy_feats + n_ally_feats * ally_id] = \
                                        obs_j[
                                        own_feats + n_enemies * n_enemy_feats + (
                                                    ally_id - 1) * n_ally_feats:own_feats + n_enemies * n_enemy_feats + n_ally_feats * (
                                            ally_id)]
                                    break
                                elif j > ally_id and obs_n[j][
                                    own_feats + n_enemies * n_enemy_feats + (ally_id - 1) * n_ally_feats] != 0:

                                    temp[i][
                                    own_feats + n_enemies * n_enemy_feats + (
                                                ally_id - 1) * n_ally_feats:own_feats + n_enemies * n_enemy_feats + n_ally_feats * ally_id] = \
                                        obs_j[
                                        own_feats + n_enemies * n_enemy_feats + (
                                            ally_id) * n_ally_feats:own_feats + n_enemies * n_enemy_feats + n_ally_feats * (
                                                    ally_id + 1)]
                                    break

            msg = temp
            obs_n = temp

            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate,
                                                          total_steps=self.total_steps)  # Get actions and the corresponding log probabilities of N agents

            totalobs = self.env.get_state()
            v_n = self.agent_n.get_value(totalobs, obs_n)  # Get the state values (V(s)) of N agents
            r, done, info = self.env.step(a_n)  # Take a step

            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                elif args.use_reward_scaling:
                    r = self.reward_scaling(r)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False
                # obs_n_orign = self.env.get_obs()
                # Store the transition



                self.replay_buffer.store_transition(episode_step, obs_n, totalobs, v_n, avail_a_n, a_n, a_logprob_n, r,
                                                    dw, msg, )

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step

            totalobs = self.env.get_state()
            # s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.args.N, 1)  # (state_dim,)-->(N,state_dim)
            v_n = self.agent_n.get_value(totalobs, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return win_tag, episode_reward, episode_step + 1

# Example of how to use this class
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(5000), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=50,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(100), help="Save frequency")
    parser.add_argument('--load_model', action='store_true', help='Whether to load a pre-trained model')
    parser.add_argument('--model_path', type=str, default=r'D:\tie-smac_IMPLUSE\model\MAPPO_env_8m_actor_number_110_seed_110_step_1000k.pth',
                        help='Path to the pre-trained model')

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False,
                        help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False,
                        help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_agent_specific", type=float, default=True,
                        help="Whether to use agent specific global state.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--device', type=str, default='cuda:0', help='whether to use the GPU')
    args = parser.parse_args()
    seeds = [3333] # 示例种子，你可以换成任何你想要的数字
    
    # 2. 定义环境
    env_names = ['8m', '1c3s5z', '2s3z']
    env_index = 0
    env_name = env_names[env_index]

    # 3. 循环遍历所有种子
    for seed in seeds:
        print("\n" + "="*80)
        print(f"=============== Running Experiment for Env: {env_name}, Seed: {seed} ===============")
        print("="*80 + "\n")


        runner = Runner_MAPPO_SMAC1(args, 
                                    env_name=env_name, 
                                    number=seed,  # 使用 seed 作为实验编号
                                    seed=seed)


        runner.run()

    print("\n\nAll experiments with different seeds are finished!")
