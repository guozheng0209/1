import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import copy

from api import call_agent_app
# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        x = self.fc2(self.rnn_hidden)

        x[avail_a_n == 0] = -1e10  # Mask the unavailable actions
        prob = torch.softmax(x, dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input, avail_a_n):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, max_episode_len, N, actor_input_dim), prob.shape(mini_batch_size, max_episode_len, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        x = self.fc3(x)
        print(avail_a_n)
        x[avail_a_n == 0] = -1e10  # Mask the unavailable actions
        prob = torch.softmax(x, dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, max_episode_len, N, critic_input_dim), value.shape=(mini_batch_size, max_episode_len, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO_SMAC:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        #orign
        self.state_dim = args.state_dim
        #self.state_dim = 160
        self.action_dim = args.action_dim
        self.args = args
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_agent_specific = args.use_agent_specific
        self.use_value_clip = args.use_value_clip
        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.obs_dim#args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N
        if self.use_agent_specific:
            print("------use agent specific global state------")
            self.critic_input_dim += args.state_dim

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
            if args.cuda:
                # self.eval_rnn = nn.DataParallel(self.eval_rnn)
                self.actor = self.actor.to(args.device)
                # self.target_rnn = nn.DataParallel(self.target_rnn)
                self.critic = self.critic.to(args.device)
        else:
            self.actor = Actor_MLP(args, self.critic_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)
            if args.cuda:
                # self.eval_rnn = nn.DataParallel(self.eval_rnn)
                self.actor = self.actor.to(args.device)
                # self.target_rnn = nn.DataParallel(self.target_rnn)
                self.critic = self.critic.to(args.device)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    # def choose_action(self, obs_n, avail_a_n, evaluate):
    #     with torch.no_grad():
    #         actor_inputs = []
    #
    #         obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
    #         actor_inputs.append(obs_n)
    #
    #         if self.add_agent_id:
    #             """
    #                 Add an one-hot vector to represent the agent_id
    #                 For example, if N=3:
    #                 [obs of agent_1]+[1,0,0]
    #                 [obs of agent_2]+[0,1,0]
    #                 [obs of agent_3]+[0,0,1]
    #                 So, we need to concatenate a N*N unit matrix(torch.eye(N))
    #             """
    #             actor_inputs.append(torch.eye(self.N))
    #
    #         actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_input.shape=(N, actor_input_dim)
    #         avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
    #
    #         # print(actor_inputs.device)
    #         # print(avail_a_n.device)
    #         if self.args.cuda:
    #             actor_inputs =actor_inputs.to(self.args.device)
    #             avail_a_n = avail_a_n.to(self.args.device)
    #
    #         prob = self.actor(actor_inputs, avail_a_n)  # prob.shape=(N, action_dim)
    #         if evaluate:  # When evaluating the policy, we select the action with the highest probability
    #             a_n = prob.argmax(dim=-1)
    #             return a_n.cpu().numpy(), None
    #         else:
    #             dist = Categorical(probs=prob)
    #             a_n = dist.sample()
    #             a_logprob_n = dist.log_prob(a_n)
    #             return a_n.cpu().numpy(), a_logprob_n.cpu().numpy()

    def choose_action(self, obs_n, avail_a_n, evaluate, total_steps):
        # 确保 obs_n 和 avail_a_n 是 NumPy 数组
        if not isinstance(obs_n, np.ndarray):
            obs_n = np.array(obs_n, dtype=np.float32) # 指定 dtype 避免 UserWarning
        if not isinstance(avail_a_n, np.ndarray):
            avail_a_n = np.array(avail_a_n, dtype=np.float32) # 指定 dtype

        if total_steps < 0:
            return self.choose_action_less_than_100k(obs_n, avail_a_n, evaluate)
        else:
            agent_0_available_actions = avail_a_n[0]
            # 判断 Agent 0 是否死亡：如果 No-op (索引0) 是1，且其他动作之和为0
            is_agent_0_dead = (agent_0_available_actions[0] == 1 and np.sum(agent_0_available_actions[1:]) == 0)

            # (可选) 更准确的死亡判断，如果 obs_n[0] 包含可解析的健康值
            # 示例: 假设 own_feats (含健康值) 在 obs_n[0] 的最后5个元素，健康值是第一个
            # own_features_start_idx = self.obs_dim - 5 # 假设 own_feats 是最后5个
            # agent_0_health = obs_n[0][own_features_start_idx]
            # if agent_0_health <= 0.001: # 考虑浮点数精度
            #     is_agent_0_dead = True
            #     print("Agent 0 confirmed dead by health feature.") # Debug

            if is_agent_0_dead:
                print("Agent 0 is dead (based on avail_actions), all agents will use PPO policy.")
                return self.choose_action_less_than_100k(obs_n, avail_a_n, evaluate)
            else:
                return self.choose_action_more_than_100k(obs_n, avail_a_n, evaluate)

    def choose_action_less_than_100k(self, obs_n_np, avail_a_n_np, evaluate):
        with torch.no_grad():
            actor_inputs_list = []

            # 不需要再次检查类型，因为在主 choose_action 中已经确保是 ndarray
            obs_n_tensor = torch.tensor(obs_n_np, dtype=torch.float32)
            actor_inputs_list.append(obs_n_tensor)

            if self.add_agent_id:
                id_tensor = torch.eye(self.N, device=obs_n_tensor.device)
                actor_inputs_list.append(id_tensor)

            actor_inputs_tensor = torch.cat(actor_inputs_list, dim=-1)
            avail_a_n_tensor = torch.tensor(avail_a_n_np, dtype=torch.float32)

            if self.args.cuda: # self.args 需要在 __init__ 中被正确设置
                actor_inputs_tensor = actor_inputs_tensor.to(self.args.device)
                avail_a_n_tensor = avail_a_n_tensor.to(self.args.device)

            prob = self.actor(actor_inputs_tensor, avail_a_n_tensor)
            if evaluate:
                a_n = prob.argmax(dim=-1)
                return a_n.cpu().numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.cpu().numpy(), a_logprob_n.cpu().numpy()

    def get_action_from_api(self, agent_id, all_agents_probs, obs_n_np, avail_a_n_np, evaluate):
        """
        辅助函数：为指定的 agent_id 获取动作。
        它会尝试调用API，如果失败，则使用PPO策略作为回退。
        """
        agent_obs_list = obs_n_np[agent_id].tolist()
        agent_avail_a_list = avail_a_n_np[agent_id].tolist()

        print(f"智能体 {agent_id} 的可用动作 (给API): {agent_avail_a_list}")

        # 你可以在这里为不同的智能体设置不同的Prompt，比如不同的性格
        # 为简化，我们暂时使用相同的Prompt
        impulse_score = 90  # 可以根据 agent_id 设定不同的值
        user_prompt_for_api = f"""
            Impulse Score: {impulse_score}
            Agent observation: {agent_obs_list}
            Agent avail action: {agent_avail_a_list}
            (此处可以放入你最终选定的完整Prompt内容)
        """

        action_text = call_agent_app(user_prompt_for_api)
        print(f"API 为智能体 {agent_id} 返回的动作文本: '{action_text}'")

        parsed_action_index = None
        if action_text:
            parsed_action_index = self.parse_action_from_text(
                action_text,
                avail_a_n_np[agent_id]
            )

        print(f"解析后的智能体 {agent_id} 的动作索引: {parsed_action_index}")

        # 如果API调用或解析失败，则使用PPO回退
        if parsed_action_index is None:
            print(f"API for Agent {agent_id} failed. Agent {agent_id} will use PPO policy.")
            agent_probs = all_agents_probs[agent_id]
            if evaluate:
                action_index = agent_probs.argmax(dim=-1).item()
                log_prob = None
            else:
                dist = Categorical(probs=agent_probs.unsqueeze(0))
                chosen_action_tensor = dist.sample()
                action_index = chosen_action_tensor.item()
                log_prob = dist.log_prob(chosen_action_tensor).item()
            return action_index, log_prob
        else:
            # API成功，计算对应的log_prob
            action_index = parsed_action_index
            log_prob = None
            if not evaluate:
                action_prob_value = all_agents_probs[agent_id, action_index].item()
                log_prob = np.log(action_prob_value) if action_prob_value > 1e-9 else -np.inf
            return action_index, log_prob

    def choose_action_more_than_100k(self, obs_n_np, avail_a_n_np, evaluate):
        with torch.no_grad():
            # --- 1. 准备工作：计算PPO概率并判断每个智能体是否死亡 ---
            agent_is_dead = []
            for i in range(self.N):
                agent_avail_actions = avail_a_n_np[i]
                is_dead = (agent_avail_actions[0] == 1 and np.sum(agent_avail_actions[1:]) == 0)
                agent_is_dead.append(is_dead)
                if is_dead:
                    print(f"智能体 {i} 已死亡（根据可用动作判断）.")

            # --- 准备 actor 输入 ---
            actor_inputs_list = [torch.tensor(obs_n_np, dtype=torch.float32)]
            if self.add_agent_id:
                id_tensor = torch.eye(self.N, device=actor_inputs_list[0].device)
                actor_inputs_list.append(id_tensor)

            actor_inputs_tensor = torch.cat(actor_inputs_list, dim=-1)
            avail_a_n_tensor = torch.tensor(avail_a_n_np, dtype=torch.float32)

            if self.args.cuda:
                actor_inputs_tensor = actor_inputs_tensor.to(self.args.device)
                avail_a_n_tensor = avail_a_n_tensor.to(self.args.device)

            all_agents_probs = self.actor(actor_inputs_tensor, avail_a_n_tensor)

            final_a_n = np.zeros(self.N, dtype=int)
            final_a_logprob_n = np.zeros(self.N, dtype=np.float32) if not evaluate else None

            # --- 2. 决策阶段 ---
            api_agent_ids = [0,1,2]  # ✅ 只为 agent0 使用 API

            for i in range(self.N):
                if i in api_agent_ids:
                    print(f"为智能体 {i} 调用 API（不论是否死亡）...")
                    action_index, log_prob = self.get_action_from_api(i, all_agents_probs, obs_n_np, avail_a_n_np,
                                                                      evaluate)
                    final_a_n[i] = action_index

                    if final_a_logprob_n is not None:
                        if log_prob is not None:
                            final_a_logprob_n[i] = log_prob
                        else:
                            prob_val = all_agents_probs[i, action_index].item()
                            final_a_logprob_n[i] = np.log(prob_val) if prob_val > 1e-9 else -np.inf
                else:
                    if agent_is_dead[i]:
                        final_a_n[i] = 0  # No-op
                        if final_a_logprob_n is not None:
                            prob_val = all_agents_probs[i, 0].item()
                            final_a_logprob_n[i] = np.log(prob_val) if prob_val > 1e-9 else -np.inf
                    else:
                        print(f"为智能体 {i} 使用 PPO 策略...")
                        agent_i_probs = all_agents_probs[i]
                        if evaluate:
                            final_a_n[i] = agent_i_probs.argmax(dim=-1).item()
                        else:
                            dist_i = Categorical(probs=agent_i_probs.unsqueeze(0))
                            chosen_action_tensor_i = dist_i.sample()
                            final_a_n[i] = chosen_action_tensor_i.item()
                            if final_a_logprob_n is not None:
                                final_a_logprob_n[i] = dist_i.log_prob(chosen_action_tensor_i).item()

            return final_a_n, final_a_logprob_n

    def parse_action_from_text(self, action_text_from_llm, avail_actions_np_array):
        action_map = {
            "无操作": 0, "no operation": 0, "noop": 0,
            "停止": 1,
            "向北移动": 2,
            "向南移动": 3,
            "向东移动": 4,
            "向西移动": 5,
            "攻击敌人[0]": 6, "attack enemy 0": 6,
            "攻击敌人[1]": 7, "attack enemy 1": 7,
            "攻击敌人[2]": 8, "attack enemy 2": 8,
            "攻击敌人[3]": 9, "attack enemy 3": 9,
            "攻击敌人[4]": 10, "attack enemy 4": 10,
            "攻击敌人[5]": 11, "attack enemy 5": 11,
            "攻击敌人[6]": 12, "attack enemy 6": 12,
            "攻击敌人[7]": 13, "attack enemy 7": 13,
            "攻击敌人[8]": 14, "attack enemy 8": 14,
        }

        if not isinstance(action_text_from_llm, str):
            print(f"解析错误: LLM 返回的不是字符串: {action_text_from_llm}")
            return None

        processed_text = action_text_from_llm.strip().lower()
        if processed_text.endswith('.'):
            processed_text = processed_text[:-1]

        print(f"用于解析的文本: '{processed_text}' (原始: '{action_text_from_llm}')")

        matched_action_index = None
        if processed_text in action_map:
            matched_action_index = action_map[processed_text]
        else:
            normalized_text_llm = processed_text.replace(" ", "").replace("[", "").replace("]", "")
            for key_map, idx_map in action_map.items():
                normalized_key_map = key_map.replace(" ", "").replace("[", "").replace("]", "")
                if normalized_key_map == normalized_text_llm:
                    matched_action_index = idx_map
                    break

        if matched_action_index is not None:
            if 0 <= matched_action_index < len(avail_actions_np_array):
                if avail_actions_np_array[matched_action_index] == 1: # NumPy array可以直接比较
                    print(f"成功解析并验证动作: '{processed_text}' -> 索引 {matched_action_index}")
                    return matched_action_index
                else:
                    print(f"解析成功但动作不可用: '{processed_text}' -> 索引 {matched_action_index}. 可用性: {avail_actions_np_array[matched_action_index]}")
                    return None
            else:
                print(f"解析成功但索引越界: '{processed_text}' -> 索引 {matched_action_index}. 可用动作长度: {len(avail_actions_np_array)}")
                return None
        else:
            print(f"无法将文本 '{processed_text}' 映射到已知动作。")
            return None


    def get_value(self, s, obs_n):
        with torch.no_grad():
            critic_inputs = []
            # critic_inputs = torch.tensor(critic_inputs, dtype=torch.float32)
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            #orign mappo
            #s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            # graph mappo

            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            #s = [s]*self.N

            s = torch.tensor(s, dtype=torch.float32)
            #print(s.shape)
            if self.args.cuda:
                s.to(self.args.device)
            critic_inputs.append(s)
            if self.use_agent_specific:  # Add local obs of agents
                critic_inputs.append(torch.tensor(obs_n, dtype=torch.float32))
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
            critic_inputs = torch.tensor(critic_inputs, dtype=torch.float32)
            if self.args.cuda:
                critic_inputs = critic_inputs.to(self.args.device)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.cpu().numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # Get training data
        max_episode_len = replay_buffer.max_episode_len

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        r = batch['r']
        v_n = batch['v_n']
        dw = batch['dw']
        active = batch['active']
        avail_a_n = batch['avail_a_n']
        a_n = batch['a_n']

        a_logprob_n = batch['a_logprob_n']

        actor_inputs, critic_inputs = self.get_inputs(batch, max_episode_len)
        if self.args.cuda:

            # self.eval_rnn = nn.DataParallel(self.eval_rnn)
            r = r.to(self.args.device)
            # self.target_rnn = nn.DataParallel(self.target_rnn)
            v_n = v_n.to(self.args.device)
            dw = dw.to(self.args.device)
            active = active.to(self.args.device)
            avail_a_n = avail_a_n.to(self.args.device)
            a_n = a_n.to(self.args.device)
            a_logprob_n = a_logprob_n.to(self.args.device)
            critic_inputs = critic_inputs.to(self.args.device)
            actor_inputs = actor_inputs.to(self.args.device)

        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len,N)
            deltas = r + self.gamma * v_n[:, 1:] * (1 - dw) - v_n[:, :-1]
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,max_episode_len,N)
            v_target = adv + v_n[:, :-1]  # v_target.shape(batch_size,max_episode_len,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv_copy = copy.deepcopy(adv)
                adv_copy = adv_copy.cpu()

                adv_copy.numpy()[batch['active'].numpy() == 0] = np.nan
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
                # adv_copy = copy.deepcopy(adv.numpy())
                # adv_copy[batch['active'].numpy() == 0] = np.nan
                # adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
                # adv_copy = copy.deepcopy(adv)
                # adv_copy[batch['active'].numpy() == 0] = np.nan
                # adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        # actor_inputs, critic_inputs = self.get_inputs(batch, max_episode_len)
        # if self.args.cuda:
        #     critic_inputs = critic_inputs.to(self.args.device)
        #     actor_inputs = actor_inputs.to(self.args.device)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    Get probs_now and values_now
                    probs_now.shape=(mini_batch_size, max_episode_len, N, action_dim)
                    values_now.shape=(mini_batch_size, max_episode_len, N)
                """
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []

                    for t in range(max_episode_len):
                        # prob.shape=(mini_batch_size*N, action_dim)

                        prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1),
                                          avail_a_n[index, t].reshape(self.mini_batch_size * self.N, -1))
                        probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))  # prob.shape=(mini_batch_size,N,action_dim）
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))  # v.shape=(mini_batch_size*N,1)
                        values_now.append(v.reshape(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size,N)
                    # Stack them according to the time (dim=1)
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    probs_now = self.actor(actor_inputs[index], avail_a_n[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, max_episode_len, N)
                # batch['a_n'][index].shape=(mini_batch_size, max_episode_len, N)
                a_logprob_n_now = dist_now.log_prob(a_n[index])  # a_logprob_n_now.shape=(mini_batch_size, max_episode_len, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - a_logprob_n[index].detach())  # ratios.shape=(mini_batch_size, max_episode_len, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = (actor_loss * active[index]).sum() / active[index].sum()

                if self.use_value_clip:
                    values_old = v_n[index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2
                critic_loss = (critic_loss * active[index]).sum() / active[index].sum()

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss + critic_loss
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch, max_episode_len):
        actor_inputs, critic_inputs = [], []




        actor_inputs.append(batch['obs_n'])


        #graph shi xiu gai
        critic_inputs.append(batch['s'])

        # #orign mappo
       # critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        # if self.args.cuda:
        #     critic_inputs = critic_inputs.to(self.args.device)
        #     actor_inputs = actor_inputs.to(self.args.device)
        if self.use_agent_specific:
            critic_inputs.append(batch['obs_n'])
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    # def load_model(self, env_name, number, seed, step):
    #     self.actor.load_state_dict(torch.load("./model/MAPPO_env_{}_actor_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))

    def load_model(self,path):
         self.actor.load_state_dict(torch.load(path))

