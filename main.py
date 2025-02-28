from sac import SAC

ENV_NAME = "Pendulum-v1" #Pendulum-v1 #LunarLanderContinuous-v2 #MountainCarContinuous-v0

agent = SAC(env_name=ENV_NAME, tau = 0.1, reward_scale=10)
agent.run()
