import matplotlib.pyplot as plt

'''
对不同参数下训练结果进行比较
'''

AVE_NUM = 300
ave_rewards = [[],[]]
rewards = [[],[]]

f1 = open('data/info.log', 'r')
f2 = open('data/info1.log', 'r')

file_list = [f1, f2]

for i, f in enumerate(file_list):
    for line in f.readlines()[1:8000]:
        line_split = line.split(' ')
        reward = float(line_split[4][6:-1])
        rewards[i].append(reward)
        if len(rewards[i]) > AVE_NUM:
            ave_rewards[i].append(sum(rewards[i][len(rewards[i])-AVE_NUM : len(rewards[i])]) / AVE_NUM)

for i, f in enumerate(file_list):
    plt.plot(ave_rewards[i])
    plt.xlabel('episodes')
    plt.ylabel('reward')

    file_list[i].close()
plt.legend(['DQN', 'DDQN', 'lr=1e-5'])
plt.show()
