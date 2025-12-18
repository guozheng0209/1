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

# replay_buffer is already imported from the other file, this can be removed if redundant.
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
        self.args.N = self.env_info["n_agents"]
        self.args.obs_dim = self.env_info["obs_shape"]
        self.args.state_dim = self.env_info["state_shape"]
        self.args.action_dim = self.env_info["n_actions"]
        self.args.episode_limit = self.env_info["episode_limit"]

        # Storing these from env_info to use in attack action check
        self.n_actions_no_attack = 6
        self.n_enemies = self.env.n_enemies

        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = MAPPO_SMAC(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Load pre-trained model if exists
        if True:  # This logic is from your original code
            print("Checking if a pre-trained model exists...")
            model_path = self.args.model_path
            if os.path.exists(model_path):
                print(f"Loading pre-trained model from {model_path}...")
                try:
                    self.agent_n.load_model(self.args.model_path)
                    print(f"Model loaded successfully from {model_path}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
            else:
                print(f"Model path {model_path} does not exist.")
        else:
            print("No pre-trained model specified for loading.")

        # Create tensorboard for logging
        self.writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.win_rates = []
        self.total_steps = 0

        # --- NEW CODE: BEHAVIORAL DATA RECORDING SETUP ---
        self.behavioral_data_container = []
        self.case_study_recorded = False  # Flag to ensure we only record and save data once
        # --- END NEW CODE ---

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

        # Final evaluation logic from your original code
        self.evaluate_policy()
        self.plt(win_rates, evaluate_rewards)

        model_name_prefix = f"{self.env_name}_seed_{self.seed}"
        self.agent_n.save_model(model_name_prefix, self.number, self.seed, self.total_steps)
        self.env.close()

    def evaluate_policy(self):
        win_times = 0
        evaluate_reward = 0
        # Your original code has `evaluate_times` as float, ensuring it's int for range
        for _ in range(int(self.args.evaluate_times)):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward

        win_rate = win_times / self.args.evaluate_times if self.args.evaluate_times > 0 else 0.0
        evaluate_reward = evaluate_reward / self.args.evaluate_times if self.args.evaluate_times > 0 else 0.0
        self.win_rates.append(win_rate)
        print(f"total_steps:{self.total_steps} \t win_rate:{win_rate} \t evaluate_reward:{evaluate_reward}")
        return win_rate, evaluate_reward

    def plt(self, win_rates, rewards):
        plt.figure()
        # Your original code uses 105, which is fine
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

        save_path_dir = os.path.join('D:', 'tie-smac_IMPLUSE', 'result', 'ppo', 'graph-1c3s5z')
        os.makedirs(save_path_dir, exist_ok=True)
        base_filename = f'env_{self.env_name}_seed_{self.seed}'
        img_save_path = os.path.join(save_path_dir, f'{base_filename}_plot.png')
        plt.savefig(img_save_path, format='png')

        win_rates_save_path = os.path.join(save_path_dir, f'{base_filename}_win_rates.npy')
        np.save(win_rates_save_path, np.array(win_rates))

        rewards_save_path = os.path.join(save_path_dir, f'{base_filename}_rewards.npy')
        np.save(rewards_save_path, np.array(rewards))

        print(f"Results for seed {self.seed} saved to {save_path_dir}")
        plt.close()

    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()

        # --- BEHAVIORAL DATA RECORDING LOGIC ---
        should_record_this_episode = evaluate and not self.case_study_recorded
        if should_record_this_episode:
            self.behavioral_data_container.clear()
            print("\n[BEHAVIORAL ANALYSIS] Recording data for ALL agents in this evaluation episode...")
        # --- END LOGIC ---

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
            if hasattr(self.agent_n.actor, 'msg_rnn_hidden'):  # Check if these attributes exist
                self.agent_n.actor.msg_rnn_hidden = None
            if hasattr(self.agent_n.critic, 'msg_rnn_hidden'):
                self.agent_n.critic.msg_rnn_hidden = None

        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()
            avail_a_n = self.env.get_avail_actions()

            # --- START: FULL GRAPH CALCULATION LOGIC (FROM YOUR ORIGINAL CODE) ---
            pos_array = self.env.get_pos_array()

            dis_array = [[] for _ in range(self.args.N)]  # Correct initialization

            G = nx.Graph()

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
                            dis_array[x].append(10000)
                else:
                    for j in range(self.args.N):
                        dis_array[x].append(10000)

            for index in range(self.args.N):
                for i in range(self.args.N):
                    if i != index and dis_array[index][i] <= pos_array[index][2] / 3 and pos_array[index][2] != 0 and \
                            pos_array[i][2] != 0:
                        G.add_edge(index, i, weight=dis_array[index][i])

            for index in range(self.args.N):
                if pos_array[index][2] != 0 and G.has_node(index) and G.degree(index) == 0:
                    min_dist = min(dis_array[index])
                    if min_dist < 10000:
                        min_index = dis_array[index].index(min_dist)
                        G.add_edge(index, min_index, weight=min_dist)

            tem_near = 1500
            node_i, node_j = 0, 0
            sub_nod_array = [list(c) for c in nx.connected_components(G)]

            if len(sub_nod_array) > 1:
                big_sub_idx = max(range(len(sub_nod_array)), key=lambda i: len(sub_nod_array[i]))
                sub_nod_array[0], sub_nod_array[big_sub_idx] = sub_nod_array[big_sub_idx], sub_nod_array[0]

                for j in range(len(sub_nod_array)):
                    for k in range(j + 1, len(sub_nod_array)):
                        tem_near = 1500
                        for l in sub_nod_array[j]:
                            for m in sub_nod_array[k]:
                                if dis_array[l][m] < tem_near:
                                    tem_near = dis_array[l][m]
                                    node_i, node_j = l, m
                        if G.has_node(node_i) and G.has_node(node_j):
                            G.add_edge(node_i, node_j, weight=tem_near)
                        node_i, node_j = 0, 0

            degree_array = [[0] * self.args.N for _ in range(self.args.N)]
            tie_strenth = [[0] * self.args.N for _ in range(self.args.N)]

            for i in range(self.args.N):
                for j in range(self.args.N):
                    if j > i:
                        if pos_array[i][2] > 0 and pos_array[j][2] > 0 and nx.has_path(G, i, j):
                            g_list = list(nx.shortest_path(G, i, j))
                            i_degree = G.degree(i)
                            j_degree = G.degree(j)
                            denominator = (i_degree + j_degree + len(g_list) - 2)
                            if denominator > 0:
                                ij_strenth = len(g_list) / denominator
                                tie_strenth[i][j] = ij_strenth
                    else:
                        tie_strenth[i][j] = tie_strenth[j][i]

            temp_strengths = [s for row in tie_strenth for s in row if s != 0]
            if not temp_strengths:
                threshold_strenth = 10
            else:
                temp_strengths.sort()
                threshold_strenth = temp_strengths[-int(len(temp_strengths) * 0.3)]

            for i in range(self.args.N):
                for j in range(self.args.N):
                    if j > i:
                        if pos_array[i][2] > 0 and pos_array[j][2] > 0:
                            if tie_strenth[i][j] < threshold_strenth:
                                degree_array[i][j] = 1
                    else:
                        degree_array[i][j] = degree_array[j][i]
                if pos_array[i][2] != 0:
                    degree_array[i][i] = 1

            temp_degrees = [G.degree(i) if pos_array[i][2] != 0 and G.has_node(i) else 0 for i in range(self.args.N)]
            if any(temp_degrees):
                max_degree_index = np.argmax(temp_degrees)
                for i in range(self.args.N):
                    if pos_array[i][2] != 0:
                        degree_array[i][max_degree_index] = 1

            self.degree_array = degree_array
            G.clear()
            # --- END: FULL GRAPH CALCULATION LOGIC ---

            # --- START: Observation Sharing Logic (FROM YOUR ORIGINAL CODE) ---
            own_feats = self.env.get_obs_own_feats_size()
            move_feats = self.env.get_obs_move_feats_size()
            n_enemies, n_enemy_feats = self.env.get_obs_enemy_feats_size()
            n_allies, n_ally_feats = self.env.get_obs_ally_feats_size()

            temp = copy.deepcopy(obs_n)
            for i in range(self.args.N):
                index = np.nonzero(np.array(self.degree_array[i]))
                for enemy_id in range(n_enemies):  # Use n_enemies from env_info
                    if np.sum(obs_n[i][
                              own_feats + enemy_id * n_enemy_feats: own_feats + (enemy_id + 1) * n_enemy_feats]) == 0:
                        for j in index[0]:
                            obs_j = self.env.get_other_obs_agent(j, i)
                            if np.sum(obs_j[own_feats + enemy_id * n_enemy_feats: own_feats + (
                                    enemy_id + 1) * n_enemy_feats]) != 0:
                                temp[i][
                                own_feats + enemy_id * n_enemy_feats: own_feats + (enemy_id + 1) * n_enemy_feats] = \
                                    obs_j[
                                    own_feats + enemy_id * n_enemy_feats: own_feats + (enemy_id + 1) * n_enemy_feats]
                                break

                for ally_id in range(self.args.N):
                    if i == ally_id: continue
                    ally_obs_idx = ally_id if ally_id < i else ally_id - 1
                    if np.sum(obs_n[i][
                              own_feats + n_enemies * n_enemy_feats + ally_obs_idx * n_ally_feats: own_feats + n_enemies * n_enemy_feats + (
                                      ally_obs_idx + 1) * n_ally_feats]) == 0:
                        for j in index[0]:
                            obs_j = self.env.get_other_obs_agent(j, i)
                            j_ally_obs_idx = ally_id if ally_id < j else ally_id - 1
                            if j != ally_id and np.sum(obs_j[
                                                       own_feats + n_enemies * n_enemy_feats + j_ally_obs_idx * n_ally_feats:own_feats + n_enemies * n_enemy_feats + (
                                                               j_ally_obs_idx + 1) * n_ally_feats]) != 0:
                                temp[i][
                                own_feats + n_enemies * n_enemy_feats + ally_obs_idx * n_ally_feats: own_feats + n_enemies * n_enemy_feats + (
                                            ally_obs_idx + 1) * n_ally_feats] = \
                                    obs_j[
                                    own_feats + n_enemies * n_enemy_feats + j_ally_obs_idx * n_ally_feats: own_feats + n_enemies * n_enemy_feats + (
                                                j_ally_obs_idx + 1) * n_ally_feats]
                                break

            msg = temp
            obs_n_for_actor = msg
            # --- END: Observation Sharing Logic ---

            a_n, a_logprob_n = self.agent_n.choose_action(obs_n_for_actor, avail_a_n, evaluate=evaluate,
                                                          total_steps=self.total_steps)

            # --- DATA COLLECTION FOR ALL AGENTS ---
            if should_record_this_episode:
                for agent_id in range(self.args.N):
                    agent_unit = self.env.get_unit_by_id(agent_id)
                    if agent_unit.health > 0:
                        x_pos, y_pos = agent_unit.pos.x, agent_unit.pos.y
                        is_attacking, attack_dist = False, -1.0
                        if agent_id == 0:
                            action = a_n[agent_id]
                            if action >= self.n_actions_no_attack:
                                target_id = action - self.n_actions_no_attack
                                if target_id in self.env.enemies:
                                    target_unit = self.env.enemies[target_id]
                                    if target_unit.health > 0:
                                        is_attacking = True
                                        dist = self.env.distance(x_pos, y_pos, target_unit.pos.x, target_unit.pos.y)
                                        attack_dist = dist
                        self.behavioral_data_container.append(
                            (agent_id, episode_step, x_pos, y_pos, is_attacking, attack_dist)
                        )
            # --- END DATA COLLECTION ---

            totalobs = self.env.get_state()
            v_n = self.agent_n.get_value(totalobs, obs_n)
            r, done, info = self.env.step(a_n)

            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if self.args.use_reward_norm:
                    r = self.reward_norm(r)
                elif self.args.use_reward_scaling:
                    r = self.reward_scaling(r)

                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                self.replay_buffer.store_transition(episode_step, obs_n, totalobs, v_n, avail_a_n, a_n, a_logprob_n, r,
                                                    dw, msg)

            if done:
                break

        # --- SAVE BEHAVIORAL DATA ---
        if should_record_this_episode:
            print("[BEHAVIORAL ANALYSIS] Saving recorded data for all agents...")
            run_type = "impulsive" if self.args.agent_0_is_impulsive else "normal"
            save_path_dir = os.path.join('behavior_data')
            os.makedirs(save_path_dir, exist_ok=True)
            filename = f'full_team_behavior_env_{self.env_name}_seed_{self.seed}_type_{run_type}.npy'
            save_path = os.path.join(save_path_dir, filename)

            trajectory_data = np.array(self.behavioral_data_container, dtype=[
                ('agent_id', 'i4'), ('step', 'i4'), ('x', 'f4'), ('y', 'f4'),
                ('is_attacking', '?'), ('attack_dist', 'f4')
            ])

            np.save(save_path, trajectory_data)
            print(f"[BEHAVIORAL ANALYSIS] Data saved to {save_path}")
            self.case_study_recorded = True
        # --- END SAVE DATA ---

        if not evaluate:
            if not done:
                totalobs = self.env.get_state()
                v_n = self.agent_n.get_value(totalobs, obs_n)
                self.replay_buffer.store_last_value(episode_step + 1, v_n)
            else:
                self.replay_buffer.store_last_value(episode_step + 1, np.zeros((self.args.N,)))

        return win_tag, episode_reward, episode_step + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(5000), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=50,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(100), help="Save frequency")
    parser.add_argument('--load_model', action='store_true', help='Whether to load a pre-trained model')
    parser.add_argument('--model_path', type=str,
                        default=r'D:\tie-smac_IMPLUSE\model\MAPPO_env_1c3s5z 1c3s5z_graph_newobs-msg-nomlp-strenth_actor_number_42_seed_88_step_1000k.pth',
                        help='Path to the pre-trained model')

    parser.add_argument("--agent_0_is_impulsive", action='store_true',
                        help="Set this flag if Agent 0 is the impulsive one (LLM-driven).")

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
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
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

    seeds = [11]
    env_names = ['8m', '1c3s5z', '2s3z']
    env_index = 1
    env_name = env_names[env_index]

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"=============== Running Experiment for Env: {env_name}, Seed: {seed} ===============")
        if args.agent_0_is_impulsive:
            print("=============== MODE: AGENT 0 IS IMPULSIVE ===============")
        else:
            print("=============== MODE: ALL AGENTS ARE NORMAL ===============")
        print("=" * 80 + "\n")

        runner = Runner_MAPPO_SMAC1(args,
                                    env_name=env_name,
                                    number=seed,
                                    seed=seed)
        runner.run()

    print("\n\nAll experiments with different seeds are finished!")