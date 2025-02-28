from torch import nn
import torch
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, input_dim, num_actions, layer_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.relu = nn.ReLU()
        self.mean = nn.Linear(layer_size, num_actions)
        self.log_std = nn.Linear(layer_size, num_actions)
        

        
    def forward(self, state):
        out = self.fc1(state)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # pour eviter les valeurs extremes

        return mean, log_std
    
    def sample_action(self, state, epsilon=1e-6):
        action_mean, action_log_std = self.forward(state)
        
        action_std = torch.exp(action_log_std)
        
        normal = torch.distributions.Normal(action_mean, action_std)
        
        action_sample = normal.rsample()
        
        squashed_action = torch.tanh(action_sample) # entre -1 et 1
        
        # Compute the log probability of the sampled action, adjusting for the tanh transformation
        log_prob = normal.log_prob(action_sample) - torch.sum(
            torch.log(1 - torch.square(squashed_action) + epsilon), 
            dim=0, keepdim=True
        )
        
        return squashed_action, log_prob


# class QNetwork(nn.Module):
#     def __init__(self, inputs_dim, hidden_size=256):
#         super(QNetwork, self).__init__()
#         self.state_net = nn.Sequential(
#             nn.Linear(inputs_dim[0], 32),  
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU()
#         )

#         self.action_net = nn.Sequential(
#             nn.Linear(inputs_dim[1], 32),  
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU()
#         )
        

#         self.combined_net = nn.Sequential(
#             nn.Linear(64, hidden_size),   
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),  
#             nn.ReLU(),
#         )

#         self.q_value_output = nn.Linear(hidden_size, 1)

#     def forward(self, state, action):
#         state_features = self.state_net(state)
#         action_features = self.action_net(action)
#         combined_features = torch.cat([state_features, action_features], dim=1)
#         combined_output = self.combined_net(combined_features)
#         q_value = self.q_value_output(combined_output)
        
#         return q_value
    
    
class QNetwork(nn.Module):
    def __init__(self, inputs_dim, hidden_size=256):
        super(QNetwork, self).__init__()

        self.state_net = nn.Sequential(
            nn.Linear(inputs_dim[0], 32),  
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.action_net = nn.Sequential(
            nn.Linear(inputs_dim[1], 32),  
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.combined_net_Q1 = nn.Sequential(
            nn.Linear(64, hidden_size),   
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Q1 output
        )

        self.combined_net_Q2 = nn.Sequential(
            nn.Linear(64, hidden_size),   
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Q2 output
        )

    def forward(self, state, action):
        state_features = self.state_net(state)
        action_features = self.action_net(action)
        combined_features = torch.cat([state_features, action_features], dim=1)

        # Compute Q1 and Q2 values
        q1_value = self.combined_net_Q1(combined_features)
        q2_value = self.combined_net_Q2(combined_features)

        return q1_value, q2_value