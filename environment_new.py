#from aircraft import *
import random
import advantage
from cache import *
from missile import *

class CombatEnv(object):
    def __init__(self, state_r=None, state_b=None):
        self.theta = None  # 敌方无人机相对我方的初始方位角（2D）
        if state_r is None:
            state_r = self._state_initialize(rand=False)
        if state_b is None:
            state_b = self._state_initialize(rand=True)
        self.aircraft_r = Aircraft(state_r)
        self.aircraft_b = Aircraft(state_b)
        self.missile1 = Missile(1)
        # 虚拟对抗的敌机，用于做和真实无人机相同的动作
        # 敌机策略生成方法：敌机搜索7个动作，选取及时收益最大的动作
        self.virtual_aircraft_b = Aircraft(state_b)

        # x_r, x_b, y_r, y_b, z_r, z_b, v_r, v_b, pitch_r, pitch_b, heading_r, heading_b
        # 状态表示为：距离，AA角，ATA角....
        self.state = []
        self.action_dim = 7
        self.state_dim = 18
        self.done = False
        self.total_steps = 0
        self.cache = Cache()
        # 控制专家系统库的决策
        self.step_num = 0
        # 当对手机和p-guide导弹距离小于1500m时，辅助机动决策,决策变量
        self.b_num = 0
        self.action_emerge = 0
        self.init_x_b = 0
        self.init_x_m = 0
        self.action_random = 0
    # 初始化敌我无人机初始状态
    def _state_initialize(self, rand=False):
        # 引入theta为了使两机初始状态相对
        if self.theta is None:
            # self.theta = random.uniform(-pi, pi)  # 敌方无人机相对我方的方位角（2D）,在小角度出现？便于学习？
            self.theta = pi / 4
        if rand is False:
            x = 5000
            y = -20000
            z = 5000
            v = 700
            heading = pi/2
            roll = ROLL_INIT
            pitch = PITCH_INIT
            state = [x, y, z, v, heading, roll, pitch]
        else:
            if self.theta >= 0:
               heading =  - pi/2
            else:
               heading = pi + self.theta
            # heading = random.uniform(-pi, pi)
            # distance_from_r = random.uniform(0.4 * DIST_INIT_MAX, 0.5 * DIST_INIT_MAX)  # 初始距离
            # distance_from_r = 10000.0  # 固定距离？？ 便于学习？？
            #distance_from_r = 8000.0 / math.cos(self.theta)
            x = 10000
            y = 30000
            z = Z_INIT
            v = V_INIT
            roll = ROLL_INIT
            pitch = PITCH_INIT
            state = [x, y, z, v, heading, roll, pitch]
        return state

    def reset(self):
        """
        初始化环境，敌我无人机状态初始化
        :return: 初状态
        """
        self.done = False
        self.total_steps = 0
        self.cache.clear()

        state_r = self._state_initialize(rand=False)
        state_b = self._state_initialize(rand=True)

        self.aircraft_r.reset(state_r)
        self.aircraft_b.reset(state_b)
        self.missile1.reset()
        self.missile1.emit_mis(state_b, self.missile1.missile_init_state)
        self.cache.push_r_state(state_r)
        self.cache.push_b_state(state_b)
        self.cache.push_missile1_state(self.missile1.missile_init_state)
        self.virtual_aircraft_b.reset(state_b)

        self.step_num = 0
        self.b_num = 0
        self.action_emerge = 0
        self.init_x_m = 0
        self.init_x_b = 0
        state_norm = self._normalize(state_r, state_b, self.missile1.missile_state)
        self.state = state_norm
        return self.state

    # 状态归一化，防止差异化过大
    def _normalize(self, state_r, state_b, state_missile):
        x_r, y_r, z_r, v_r, heading_r, roll_r, pitch_r = state_r
        x_b, y_b, z_b, v_b, heading_b, roll_b, pitch_b = state_b
        x_m, y_m, z_m, v_m, vm_x, vm_y, vm_z = state_missile
        x_r = x_r / 10000.0
        x_b = x_b / 10000.0
        x_m = x_m / 10000.0
        y_r = y_r / 30000.0
        y_b = y_b / 30000.0
        y_m = y_m / 30000.0
        v_r = (v_r - 700) / 50
        v_b = (v_b - 250) / 50

        z_r = (z_r - Z_MIN) / (Z_MAX - Z_MIN)
        z_b = (z_b - Z_MIN) / (Z_MAX - Z_MIN)
        z_m = (z_m - Z_MIN) / (Z_MAX - Z_MIN)
        pitch_r = pitch_r / PITCH_MAX
        pitch_b = pitch_b / PITCH_MAX
        # roll_r = roll_r / ROLL_MAX
        # roll_b = roll_b / ROLL_MAX
        vm_x = vm_x / 400
        vm_y = vm_y / 400
        vm_z = vm_z / 400
        return [x_r, x_b, y_r, y_b, z_r, z_b, v_r, v_b, pitch_r, pitch_b, heading_r, heading_b, x_m, y_m, z_m, vm_x, vm_y, vm_z]

    # 态势评估，由敌我无人机状态解算出距离、威胁角等
    def _situation(self, state_r, state_b, state_messile1):
        x_r, y_r, z_r, v_r, heading_r, roll_r, pitch_r = state_r
        x_b, y_b, z_b, v_b, heading_b, roll_b, pitch_b = state_b
        x_m, y_m, z_m, v_m, vm_x, vm_y, vm_z = state_messile1
        # 距离向量
        vector_d = np.array([x_m - x_r, y_m - y_r, z_m - z_r])
        # 敌我无人机的速度向量
        vector_vr = np.array([math.cos(pitch_r) * math.cos(heading_r),
                              math.sin(heading_r) * math.cos(pitch_r), math.sin(pitch_r)])

        vector_vb = np.array([math.cos(pitch_b) * math.cos(heading_b),
                              math.sin(heading_b) * math.cos(pitch_b), math.sin(pitch_b)])
        vector_vm = np.array([vm_x, vm_y, vm_z])
        vector_vm_xy = np.array([vm_x, vm_y, 60])
        # AA角和ATA角计算，向量夹角
        # AA和ATA搞反了，和论文刚好相反
        aspect_angle = self._cal_angle(vector_vr, vector_d)
        coop_angle = self._cal_angle(vector_vr, vector_vm_xy)
        antenna_train_angle = self._cal_angle(vector_vr, vector_vm)
        # print("AA:{0}, ATA:{1}".format(aspect_angle, antenna_train_angle))
        #print(coop_angle)
        #print(vector_vr*v_r)
        #print(self.step_num)
        distance = np.sqrt(np.sum(vector_d * vector_d))
        return [distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b, z_m, coop_angle]

    def p_b_situation(self, state_missile, state_b):
        x_m, y_m, z_m, v_m, v_x, v_y, v_z = state_missile
        x_b, y_b, z_b, v_b, heading_b, roll_b, pitch_b = state_b

        # 距离向量
        vector_d = np.array([x_b - x_m, y_b - y_m, z_b - z_m])
        # 敌我无人机的速度向量
        vector_vr = np.array([v_x, v_y, v_z])
        vector_vb = np.array([math.cos(pitch_b) * math.cos(heading_b),
                              math.sin(heading_b) * math.cos(pitch_b), math.sin(pitch_b)])
        # AA角和ATA角计算，向量夹角
        # AA和ATA搞反了，和论文刚好相反
        aspect_angle = self._cal_angle(vector_vr, vector_d)
        antenna_train_angle = self._cal_angle(vector_vb, vector_d)
        # print("AA:{0}, ATA:{1}".format(aspect_angle, antenna_train_angle))

        distance = np.sqrt(np.sum(vector_d * vector_d))
        return [distance, aspect_angle, antenna_train_angle, z_m, z_b, v_m, v_b, pitch_b, pitch_b, roll_b, roll_b]

    def _cal_angle(self, vector_vr, vector_d):
        dot_product = np.dot(vector_vr, vector_d)
        vr_norm = np.sqrt(np.sum(vector_vr * vector_vr))
        d_norm = np.sqrt(np.sum(vector_d * vector_d))
        angle = np.arccos(dot_product / (d_norm*vr_norm + 1e-5))
        return angle

    def _cal_reward(self, situation, save=True):
        angle_reward = advantage.angle_adv(situation)
        height_reward = advantage.height_adv(situation)
        velocity_reward = advantage.velocity_adv(situation)
        dis_reward = advantage.dis_adv(situation)
        pre_angle_reward = advantage.pre_angle(situation,self.step_num)
        coop_angle_reward = advantage.coop_angle_adv(situation)
        if save is True:
            self.cache.push_angle_adv(angle_reward)
            self.cache.push_height_adv(height_reward)
            self.cache.push_velocity_adv(velocity_reward)
            self.cache.push_dis_adv(dis_reward)
            self.cache.push_pre_angle_adv(pre_angle_reward)
            self.cache.push_coop_angle_adv(coop_angle_reward)
            self.cache.push_reward(0.15*velocity_reward+0.85*pre_angle_reward+0.*dis_reward+0.*height_reward+0.*coop_angle_reward)

        #return 0.7 * angle_reward + 0.2 * height_reward + 0.1 * velocity_reward

        return 0.15*velocity_reward+0.85*pre_angle_reward+0.*dis_reward+0.*height_reward+0.*coop_angle_reward

    def _enemy_ai(self):
        """
        敌机策略生成，滚动时域法，搜索7个动作中使我方无人机回报最小的动作执行
        :return:
        """
        virtual_rewards = []
        initial_state_b = self.virtual_aircraft_b.state
        for i in range(self.action_dim):
            self.virtual_aircraft_b.maneuver(i)
            virtual_situation = self.p_b_situation(self.missile1.missile_state, self.virtual_aircraft_b.state)
            #virtual_reward = 0.7 * advantage.angle_adv(virtual_situation) + 0.2 * advantage.height_adv(
                #virtual_situation) + \
                             #0.1 * advantage.velocity_adv(virtual_situation)

            virtual_reward =0.*advantage.dis_adv(virtual_situation) +0.2*advantage.height_adv(virtual_situation) +0.6 *advantage.pre_angle(virtual_situation) +0.2*advantage.velocity_adv(virtual_situation)

            virtual_rewards.append(virtual_reward)
            # 模拟完一轮之后将状态复原
            self.virtual_aircraft_b.reset(initial_state_b)
        # 选取使得敌方虚拟收益最小的动作
        action = virtual_rewards.index(min(virtual_rewards))
        return action
        # 敌方固定策略？
        # return 0

    def _enemy_ai_2(self):
        virtual_dis = []
        initial_state_b = self.virtual_aircraft_b.state
        for i in range(self.action_dim):
            self.virtual_aircraft_b.maneuver(i)
            virtual_situation = self._situation(self.virtual_aircraft_b.state, self.aircraft_r.state)
            distance = virtual_situation[0]
            virtual_dis.append(distance)
            # 模拟完一轮之后将状态复原
            self.virtual_aircraft_b.reset(initial_state_b)
        # 选取使得敌方虚拟收益最小的动作
        action = virtual_dis.index(min(virtual_dis))
        return action

    def _enemy_ai_expert(self):
        """
        专家系统策略
        :return:
        """
        state_missile, state_b = self.missile1.missile_state, self.aircraft_b.state
        x_m, y_m, z_m, v_m, v_x, v_y, v_z = state_missile
        x_b, y_b, z_b, v_b, heading_b, roll_b, pitch_b = state_b

        # 距离向量
        vector_d = np.array([x_b - x_m, y_b - y_m, z_b - z_m])
        # 敌我无人机的速度向量
        vector_vm = np.array([v_x, v_y, v_z])
        vector_vb = np.array([math.cos(pitch_b) * math.cos(heading_b),
                              math.sin(heading_b) * math.cos(pitch_b), math.sin(pitch_b)])
        distance = np.sqrt(np.sum(vector_d * vector_d))
        # AA角和ATA角计算，向量夹角
        aspect_angle = self._cal_angle(vector_vm, vector_d)
        antenna_train_angle = self._cal_angle(vector_vb, vector_d)

        if self.step_num == 1:
            self.init_x_m = x_m
            self.init_x_b = x_b
        left_or_right = np.sign(vector_vm[0] * vector_vb[1] - vector_vm[1] * vector_vb[0])
        linear_adv = 1 - (aspect_angle + antenna_train_angle) / pi
        # 0.定常飞行； 1.加速； 2.减速； 3.左转弯； 4.右转弯； 5.拉起； 6.俯冲
        if self.step_num < 50:
            return 1
        #else:
            #if random.random() >= 0.1:
                #return self._chase(vector_vm, vector_vb, distance, z_m, z_b, self.init_x_m, self.init_x_b)
            #else:
                #if z_m > z_b:
                    #return 6
                #else:
                    #return 5
        else:

            if self.step_num%5 == 0:
                self.action_random = random.randint(0, 6)
            return self.action_random

    def _escape(self, left_or_right):
        if left_or_right > 0:
            return 4
        elif left_or_right < 0:
            return 4
        else:
            return 6

    def _chase(self, vector_vm, vector_vb, distance, z_m, z_b, init_x_m, init_x_b):
        #print(self._cal_angle(vector_vm, vector_vb))
        if distance>1500:
            if (self._cal_angle(vector_vm, vector_vb) > 2*pi/3)or(self._cal_angle(vector_vm, vector_vb) < pi/3):
                if random.random() >= 0.5:
                    return 3
                else:
                    return 4

            else:
                virtual_distances=[]
                initial_state_b = self.virtual_aircraft_b.state
                for i in range(self.action_dim):
                    self.virtual_aircraft_b.maneuver(i)
                    virtual_situation = self.p_b_situation(self.missile1.missile_state, self.virtual_aircraft_b.state)
                    virtual_distances.append(virtual_situation[0])
                    # 模拟完一轮之后将状态复原
                    self.virtual_aircraft_b.reset(initial_state_b)
                # 选取使得敌方虚拟收益最小的动作
                action = virtual_distances.index(max(virtual_distances))
                return action
        else:
            if self.b_num == 0:
                if z_m < z_b:
                    self.action_emerge = 6
                else:
                    self.action_emerge = 5
                self.b_num +=1
                return self.action_emerge
            else:
                return self.action_emerge
    def step(self, action):
        """
        执行状态的一个时间步的更新
        :param action: 执行动作
        :return: 下一状态（归一化后）、奖励、该幕是否结束
        """
        self.step_num +=1
        action_r = action
        action_b = self._enemy_ai_expert()

        state_r = self.aircraft_r.maneuver_m(action_r)
        state_b = self.aircraft_b.maneuver(action_b)
        state_missile1 = self.missile1.p_guide_sim(state_b)
        # print("b action is {0}, state is {1}".format(action_b, state_b))
        self.cache.push_r_action(action_r)
        self.cache.push_b_action(action_b)
        self.cache.push_r_state(state_r)
        self.cache.push_b_state(state_b)
        self.cache.push_missile1_state(state_missile1)

        self.virtual_aircraft_b.maneuver(action_b)

        self.state = self._normalize(state_r, state_b, state_missile1)
        situation = self._situation(state_r, state_b, state_missile1)

        reward = self._cal_reward(situation, save=True)
        self.total_steps += 1
        # self.cache.save_combat_log(state_r,state_b,reward)

        distance, z_r, aa, ata = situation[0], situation[3], situation[1], situation[2]
        # 超出近战范围或步长过大
        if self.done is False and self.total_steps >= 290:
            self.done = True

        if distance > DIST_INIT_MAX or distance < 500:
            reward = 10
            self.cache.push_reward(reward)
            self.done = True

        #if aa * rad2deg < 30 and ata * rad2deg < 45:
        #    reward = 30
        #    self.cache.push_reward(reward)
        #    self.done = True

        # if aa*rad2deg > 145 and ata*rad2deg > 150:
        #     self.done = True
        #     reward = -30
        return self.state, reward, self.done

    def get_cache(self):
        return self.cache
