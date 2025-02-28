from actor_critic import QNetwork, Actor
from replay_memory import ReplayMemory
import torch
import gym
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optim


# class SAC:
#     def __init__(self, env_name, n_epochs=100, batch_size=256, alpha=1, max_mem_length=1e6, tau=0.1,
#                  Q_lr=1e-3, policy_lr=1e-3, alpha_lr=1e-3, gamma=0.9, reward_scale = 1):
#         self.env_name = env_name
#         self.env = gym.make(self.env_name)
#         self.n_epochs = int(n_epochs)
#         self.steps = 0
#         self.total_scores = []
#         self.batch_size = batch_size
#         self.tau = tau
#         self.ReplayMemory = ReplayMemory(int(max_mem_length))
#         self.gamma = gamma
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu")
#         self.reward_scale = reward_scale

#         self.Actor = Actor(
#             self.env.observation_space.shape[0], self.env.action_space.shape[0])
#         self.Q1 = QNetwork(
#             [self.env.observation_space.shape[0], self.env.action_space.shape[0]])
#         self.Q2 = QNetwork(
#             [self.env.observation_space.shape[0], self.env.action_space.shape[0]])
#         self.target_Q1 = QNetwork(
#             [self.env.observation_space.shape[0], self.env.action_space.shape[0]])
#         self.target_Q1.load_state_dict(self.Q1.state_dict())
#         self.target_Q2 = QNetwork(
#             [self.env.observation_space.shape[0], self.env.action_space.shape[0]])
#         self.target_Q2.load_state_dict(self.Q2.state_dict())
#         self.alpha = torch.tensor(alpha, requires_grad=True)
#         self.entropy_target = torch.scalar_tensor(
#             -self.env.action_space.shape[0], dtype=torch.float64)
#         self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=Q_lr)
#         self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=Q_lr)
#         self.Actor_optimizer = optim.Adam(
#             self.Actor.parameters(), lr=policy_lr)
#         self.alpha_optimizer = optim.Adam([self.alpha], lr=alpha_lr)

#     def run(self):
#         for ep in range(self.n_epochs):
#             print(f'=========== Epoch n°{ep}==============\n')
#             total_score = 0
#             done = False
#             state = self.env.reset()[0]
#             current_step = 0
#             while (done is False) and (current_step < 300):
#                 current_step += 1
#                 self.steps += 1
#                 state = torch.tensor(state, dtype=torch.float32)
#                 action, log_pi = self.Actor.sample_action(state)
#                 action_flatten = np.squeeze(action, axis=0)
#                 env_action = torch.tensor(self.env.action_space.low, dtype=torch.float32) + \
#                     (action_flatten + 1) / 2 * \
#                     (torch.tensor(self.env.action_space.high, dtype=torch.float32) -
#                      torch.tensor(self.env.action_space.low, dtype=torch.float32))
#                 next_state, reward, done, info, _ = self.env.step(
#                     env_action.numpy())
#                 total_score += reward
#                 self.ReplayMemory.push((state.numpy(), action.numpy(),
#                                 reward, next_state, 1 - done))
#                 state = next_state
#                 if len(self.ReplayMemory.memory) > self.batch_size:
#                     self.learn()
#             self.total_scores.append(total_score)
#         # convert to pandas dataframe and save to csv
#         df = pd.DataFrame(self.total_scores, columns=['Scores'])
#         df.to_csv(f'scores_tau_{self.tau}_{self.env_name}.csv', index=False)
        
#         self.env.close()

#     def learn(self):
#         # Sample a batch of experiences from the replay buffer
#         batch = self.ReplayMemory.sample(self.batch_size)
#         state, action, reward, next_state, not_done = zip(*batch)

#         state = torch.tensor(state, dtype=torch.float32, requires_grad=True).to(self.device)
#         action = torch.tensor(action, dtype=torch.float32, requires_grad=True).to(self.device)
#         action = action.view(-1, self.env.action_space.shape[0])  
#         reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True).to(self.device)
#         reward = reward.view(-1, 1)  
#         next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
#         not_done = torch.tensor(not_done, dtype=torch.float32).to(self.device)
#         not_done = not_done.view(-1, 1)  

#         # Compute target Q-values for Q1 and Q2
#         with torch.no_grad():
#             next_action, next_log_pi = self.Actor.sample_action(next_state)
#             next_Q1 = self.target_Q1(next_state, next_action)
#             next_Q2 = self.target_Q2(next_state, next_action)
#             next_Q = torch.min(next_Q1, next_Q2) - self.alpha * next_log_pi
#             target_Q = reward * self.reward_scale + self.gamma * not_done * next_Q  # Scale reward

#         # Update Q1 and Q2 networks
#         self.Q1_optimizer.zero_grad()
#         self.Q2_optimizer.zero_grad()
#         Q1 = self.Q1(state, action)
#         Q2 = self.Q2(state, action)
#         Q1_loss = F.mse_loss(Q1, target_Q)
#         Q2_loss = F.mse_loss(Q2, target_Q)
#         Q1_loss.backward()
#         Q2_loss.backward()
#         self.Q1_optimizer.step()
#         self.Q2_optimizer.step()

#         # Update the policy network
#         self.Actor_optimizer.zero_grad()
#         sampled_action, log_pi = self.Actor.sample_action(state)
#         Q1 = self.Q1(state, sampled_action)
#         Q2 = self.Q2(state, sampled_action)
#         Q = torch.min(Q1, Q2)
#         policy_loss = (-self.alpha.detach() * log_pi - Q).mean()  # Policy loss with entropy regularization
#         policy_loss.backward()
#         self.Actor_optimizer.step()

#         # Update the temperature parameter (alpha)
#         self.alpha_optimizer.zero_grad()
#         alpha_loss = -(self.alpha * (-log_pi + self.entropy_target)).mean()  # Alpha loss
#         alpha_loss.backward(inputs=[self.alpha])
#         self.alpha_optimizer.step()


#         self.update_targets()

#     def update_targets(self):
#         for target_param, online_param in zip(self.target_Q1.parameters(), self.Q1.parameters()):
#             target_param.data.copy_(
#                 self.tau * online_param.data + (1 - self.tau) * target_param.data
#             )

#         for target_param, online_param in zip(self.target_Q2.parameters(), self.Q2.parameters()):
#             target_param.data.copy_(
#                 self.tau * online_param.data + (1 - self.tau) * target_param.data
#             )


class SAC:
    def __init__(self, env_name, n_epochs=100, batch_size=256, alpha=1.0, max_mem_length=1e6, tau=0.1,
                 Q_lr=3e-4, policy_lr=3e-4, alpha_lr=3e-4, gamma=0.9, reward_scale = 1):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.n_epochs = int(n_epochs)
        self.steps = 0
        self.total_scores = []
        self.batch_size = batch_size
        self.tau = tau
        self.ReplayMemory = ReplayMemory(int(max_mem_length))
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Actor = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.Q = QNetwork([self.env.observation_space.shape[0], self.env.action_space.shape[0]])
        self.target_Q = QNetwork([self.env.observation_space.shape[0], self.env.action_space.shape[0]])
        self.target_Q.load_state_dict(self.Q.state_dict())  

        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=Q_lr)
        self.Actor_optimizer = optim.Adam(self.Actor.parameters(), lr=policy_lr)
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.entropy_target = torch.scalar_tensor(-self.env.action_space.shape[0], dtype=torch.float64)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=alpha_lr)
        


    def run(self):
        for ep in tqdm(range(self.n_epochs), desc="Training Progress", unit="epoch"):
            total_score = 0
            done = False
            state = self.env.reset()[0] 
            current_step = 0

            while not done and current_step < 300:
                current_step += 1
                self.steps += 1

                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, log_pi = self.Actor.sample_action(state_tensor)

                action_flatten = np.squeeze(action, axis=0)
                env_action = torch.tensor(self.env.action_space.low, dtype=torch.float32) + \
                            (action_flatten + 1) / 2 * \
                            (torch.tensor(self.env.action_space.high, dtype=torch.float32) -
                            torch.tensor(self.env.action_space.low, dtype=torch.float32))

                next_state, reward, done, info, _ = self.env.step(env_action.detach().numpy())
                total_score += reward

                self.ReplayMemory.push((state, action.detach().numpy(), reward, next_state, 1 - done))

                state = next_state

                if len(self.ReplayMemory.memory) > self.batch_size:
                    self.learn()

            self.total_scores.append(total_score)

        # save the scores to a CSV file
        df = pd.DataFrame(self.total_scores, columns=['Scores'])
        df.to_csv(f'scores_tau_{self.tau}_scale_{self.reward_scale}_{self.env_name}.csv', index=False)

        self.env.close()

    def learn(self):
        batch = self.ReplayMemory.sample(self.batch_size)
        state, action, reward, next_state, not_done = zip(*batch)

        state = torch.tensor(state, dtype=torch.float32, requires_grad=True)
        action = torch.tensor(action, dtype=torch.float32, requires_grad=True)
        action = action.view(-1, self.env.action_space.shape[0])  
        reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True)
        reward = reward.view(-1, 1)  
        next_state = torch.tensor(next_state, dtype=torch.float32)
        not_done = torch.tensor(not_done, dtype=torch.float32)
        not_done = not_done.view(-1, 1)  

        # Compute target Q-values for Q1 and Q2
        with torch.no_grad():
            next_action, next_log_pi = self.Actor.sample_action(next_state)
            next_Q1, next_Q2 = self.target_Q(next_state, next_action)
            next_Q = torch.min(next_Q1, next_Q2) - self.alpha * next_log_pi
            target_Q = reward * self.reward_scale + self.gamma * not_done * next_Q  # Scale reward by 10

        # Update Q1 and Q2 networks
        self.Q_optimizer.zero_grad()
        Q1, Q2 = self.Q(state, action)
        Q1_loss = F.mse_loss(Q1, target_Q)
        Q2_loss = F.mse_loss(Q2, target_Q)
        total_Q_loss = Q1_loss + Q2_loss
        total_Q_loss.backward()
        self.Q_optimizer.step()

        # Update the policy network
        self.Actor_optimizer.zero_grad()
        sampled_action, log_pi = self.Actor.sample_action(state)
        Q1, Q2 = self.Q(state, sampled_action)
        Q = torch.min(Q1, Q2)
        policy_loss = (-self.alpha.detach() * log_pi - Q).mean()  # Policy loss with entropy regularization
        policy_loss.backward()
        self.Actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss = -(self.alpha * (-log_pi + self.entropy_target)).mean()  # Alpha loss
        alpha_loss.backward(inputs=[self.alpha])
        self.alpha_optimizer.step()

        self.update_targets()

    def update_targets(self):
        for target_param, online_param in zip(self.target_Q.parameters(), self.Q.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )