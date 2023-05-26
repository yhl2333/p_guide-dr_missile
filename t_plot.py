import matplotlib.pyplot as plt

'''
对比三种方法奖励图
'''
AVE_NUM =700
ave_rewards = []
ave_rewards1 = []
rewards = []
losses = []
ave_loss = []
ave_Q = []
Q_sum = []
f = open('data/iinfo.log', 'r')
f1 = open('data/iinfo1.log', 'r')
f2 = open('data/iinfoDuel.log', 'r')
file_list = [f, f1]
color = ['r', 'g', 'b']
for i, f in enumerate(file_list):
    for line in f.readlines()[1:20000]:
        line_split = line.split(' ')
        reward = float(line_split[3][13:])
        loss = float(line_split[4][6:-1])
        Q = float(line_split[6][6:-1])
        rewards.append(reward)
        losses.append(loss)
        Q_sum.append(Q)
        if len(rewards) > AVE_NUM:
            ave_rewards.append(sum(rewards[len(rewards)-AVE_NUM : len(rewards)]) / AVE_NUM)
            ave_loss.append(sum(losses[len(rewards) - AVE_NUM : len(rewards)]) / AVE_NUM)
            ave_Q.append(sum(Q_sum[len(rewards) - AVE_NUM : len(rewards)]) / AVE_NUM)


    # plt.subplot(121)
    # plt.plot(rewards)
    # plt.plot(ave_rewards,'r', label='original', linewidth='0.05')
    plt.plot(ave_loss, color[i], linewidth='1.5')
    rewards.clear()
    losses.clear()
    Q_sum.clear()
    ave_rewards.clear()
    ave_loss.clear()
    ave_Q.clear()

plt.xlabel('episodes')
plt.ylabel('loss')
plt.legend(['DQN', 'DDQN'])

plt.show()
f.close()
