from DQNAgent import *
from environment_new import *
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

'''
训练模型评估，读取训练好的数据进行评估绘图，动态三维对抗图，动态角度、高度、速度优势
'''

DRAW_WITH_ADVANTAGE = True

if __name__ == '__main__':
    env = CombatEnv()
    agent = Agent(env)
    agent.load_model('model/airCom24001')
    cache = agent.test_result()
    r_states = cache.get_r_states()
    b_states = cache.get_b_states()
    missile1_states = cache.get_missile1_states()
    r_actions = cache.get_r_actions()
    b_actions = cache.get_b_actions()
    rewards = cache.get_rewards()


    print("total steps:{0}; rewards is {1}".format(len(rewards), sum(rewards)))
    print(rewards)
    print(r_actions)
    print(b_actions)
    angle_adv = cache.get_angle_adv()
    height_adv = cache.get_height_adv()
    velocity_adv = cache.get_velocity_adv()
    pre_angle_adv = cache.get_pre_angle_adv()
    coop_angle_adv = cache.get_coop_angle_adv()
    dis_adv = cache.get_dis_adv()

    r_states = list(zip(*r_states))
    r_states_x = r_states[0]
    r_states_y = r_states[1]
    r_states_z = r_states[2]
    r_states_v = r_states[3]

    b_states = list(zip(*b_states))
    b_states_x = b_states[0]
    b_states_y = b_states[1]
    b_states_z = b_states[2]
    b_states_v = b_states[3]

    missile1_states = list(zip(*missile1_states))
    missile1_states_x = missile1_states[0]
    missile1_states_y = missile1_states[1]
    missile1_states_z = missile1_states[2]
    missile1_states_v = missile1_states[3]


    if DRAW_WITH_ADVANTAGE:
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(121, projection='3d')
        plt.title('AirCombat')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_zlabel('z(m)')

        ax1 = fig1.add_subplot(243)
        plt.title("total advantage")
        ax1.set_xlabel('steps')
        ax1.set_ylabel('rewards')

        ax2 = fig1.add_subplot(244)
        plt.title("Real-time pre_angle advantage")
        ax2.set_xlabel('steps')
        ax2.set_ylabel('angle advantage')

        ax3 = fig1.add_subplot(247)
        plt.title("Real-time velocity advantage")
        ax3.set_xlabel('steps')
        ax3.set_ylabel('dis advantage')

        ax4 = fig1.add_subplot(248)
        plt.title("Real-time coop_angle advantage")
        ax4.set_xlabel('steps')
        ax4.set_ylabel('coop advantage')

        ax.plot(r_states_x[:1], r_states_y[:1], r_states_z[:1], 'g', marker='o', markersize=10, label='start')
        ax.plot(b_states_x[:1], b_states_y[:1], b_states_z[:1], 'g', marker='o', markersize=10)
        ax.plot(missile1_states_x[:1], missile1_states_y[:1], missile1_states_z[:1], 'g', marker='o', markersize=10)
        for i in range(len(r_states_x)):
            ax.plot(r_states_x[0:i], r_states_y[0:i], r_states_z[0:i], 'r')
            ax.plot(b_states_x[0:i], b_states_y[0:i], b_states_z[0:i], 'b')
            ax.plot(missile1_states_x[0:i], missile1_states_y[0:i], missile1_states_z[0:i], 'k')
            ax1.plot(rewards[0:i], 'b')
            ax2.plot(pre_angle_adv[0:i], 'b')
            ax3.plot(velocity_adv[0:i], 'b')
            ax4.plot(coop_angle_adv[0:i], 'b')
            plt.pause(0.05)
        ax.plot(r_states_x, r_states_y, r_states_z, 'r', label='aircraft_r')
        ax.plot(b_states_x, b_states_y, b_states_z, 'b', label='aircraft_b')
        ax.plot(missile1_states_x, missile1_states_y, missile1_states_z, 'k', label='p-guide')

        ax.plot(r_states_x[-1:], r_states_y[-1:], r_states_z[-1:], 'black', marker='x', markersize=10, label='end')
        ax.plot(b_states_x[-1:], b_states_y[-1:], b_states_z[-1:], 'black', marker='x', markersize=10)
        ax.plot(missile1_states_x[-1:], missile1_states_y[-1:], missile1_states_z[-1:], 'black', marker='x',
                markersize=10)
        ax.legend(loc='upper right')
        plt.show()
    else:
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(projection='3d')
        plt.title('AirCombat')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_zlabel('z(m)')

        ax.plot(r_states_x[:1], r_states_y[:1], r_states_z[:1], 'g', marker='o', markersize=10, label='start')
        ax.plot(b_states_x[:1], b_states_y[:1], b_states_z[:1], 'g', marker='o', markersize=10)
        ax.plot(missile1_states_x[:1], missile1_states_y[:1], missile1_states_z[:1], 'g', marker='o', markersize=10)
        for i in range(len(r_states_x)):
            ax.plot(r_states_x[0:i], r_states_y[0:i], r_states_z[0:i], 'r')
            ax.plot(b_states_x[0:i], b_states_y[0:i], b_states_z[0:i], 'b')
            ax.plot(missile1_states_x[0:i], missile1_states_y[0:i], missile1_states_z[0:i], 'k')
            plt.pause(0.05)
        ax.plot(r_states_x, r_states_y, r_states_z, 'r', label='missile_r')
        ax.plot(b_states_x, b_states_y, b_states_z, 'b', label='aircraft_b')
        ax.plot(missile1_states_x, missile1_states_y, missile1_states_z, 'k', label='p-guide')

        ax.plot(r_states_x[-1:], r_states_y[-1:], r_states_z[-1:], 'black', marker='x', markersize=10, label='end')
        ax.plot(b_states_x[-1:], b_states_y[-1:], b_states_z[-1:], 'black', marker='x', markersize=10)
        ax.plot(missile1_states_x[-1:], missile1_states_y[-1:], missile1_states_z[-1:], 'black', marker='x', markersize=10)
        ax.legend(loc='upper right')
        plt.show()
