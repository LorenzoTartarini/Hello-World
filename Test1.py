import gym
import numpy as np

env = gym.make('CartPole-v0')
LEARNING_RATE= 0.1
SHOW_EVERY = 5000
DISCOUNT= 0.9 #measure of how we value next q wrt to actual q, from 0 to 1
EPISODES = 200000
ANGLE_MAX= 41
VEL_MAX= 10000
MAX=np.array([ANGLE_MAX,VEL_MAX])
MIN=-MAX


epsilon = 0.1 #random action to explore stuff
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# DISCRETE_OS_SIZE = [20]*len(env.observation_space.high) #20x20
# print(f"D# ISCRETE OS SIZE= {[20]*len(env.observation_space.high)} ")
# discrete_os_win_size = 2*env.observation_space.high/DISCRETE_OS_SIZE
# print(discrete_os_win_size)


n_features=2
DISCRETE_OS_SIZE=[20]*n_features
discrete_os_win_size = 2*MAX/DISCRETE_OS_SIZE
#print(f"discrete_os_win_size= {discrete_os_win_size}")
maxCount=0
count=0
q_table=np.random.uniform(low=0,high=2, size=(DISCRETE_OS_SIZE + [env.action_space.n]))#20x20x3 with initial random Q values between -2 and 0
#print(q_table.shape)

def get_discrete_state(state):
    discrete_state = (state - MIN)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for i_episode in range(EPISODES):

	if i_episode%SHOW_EVERY ==0:
		render=True
	else:
		render=False
	done=False
	obs = env.reset()
	#state = tuple(obs.astype(np.int))
	state = get_discrete_state(obs[2:])

	if count>maxCount:
		maxCount=count
	count=0
	while not done:
		if np.random.random() > epsilon:
			action=np.argmax(q_table[state]) #exploitation
		else:
			action=np.random.randint(0,env.action_space.n) #exploration
		count+=1
		new_obs,reward,done,_ = env.step(action)
		new_state=get_discrete_state(obs[2:])
		#new_state=tuple(new_obs.astype(np.int))
		if not done:
			max_future_q= np.max(q_table[new_state])
			current_q= q_table[state + (action, )]
			new_q=(1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT* max_future_q)
			q_table[state + (action, )] = new_q
		elif count>= 100:
			q_table[state + (action, )] =0
			print(f"We made it on episode {episode}")
		state=new_state
		if render:
			env.render()
	if END_EPSILON_DECAYING >= i_episode >= START_EPSILON_DECAYING:
		epsilon-= epsilon_decay_value
	#env.render()
	print(count)
print(maxCount)
env.close()