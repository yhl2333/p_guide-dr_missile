import math

import numpy as np

from config import *

AA_MAX = 180 * deg2rad
ATA_MAX = 180 * deg2rad


# 计算我方对敌方的角度优势、高度优势及速度优势
# 参数均为归一化之前的飞机状态参数
def angle_adv(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b, z_m, coop_angle = state
    return 0.5*(_angle_adv(aspect_angle, antenna_train_angle) - _angle_adv(pi - antenna_train_angle, pi - aspect_angle))

def angle_adv_half(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b, z_m, coop_angle = state
    if aspect_angle <= AA_MAX:
        adv_r = math.exp(-aspect_angle / AA_MAX)
    else:
        adv_r = 0
    if antenna_train_angle >= pi - ATA_MAX:
        adv_b = - math.exp(-(pi - antenna_train_angle) / ATA_MAX)
    else:
        adv_b = 0
    return adv_r + adv_b
def angle_adv_linear(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b, z_m, coop_angle = state
    return 1 - (aspect_angle + antenna_train_angle) / pi

def height_adv(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b, z_m, coop_angle = state
    if z_r - z_b > 1000:
        return -1.0
    elif z_r - z_b < -1000:
        return -1.0
    else:
        return (1000-np.abs(z_r - z_b)) / 1000.0

def dis_adv(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b, z_m, coop_angle = state
    return 2*math.exp(-distance/50000)-1

def pre_angle(state,step_num):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b, z_m, coop_angle = state
    if 220 > step_num > 130:
        if coop_angle < pi/5:
            adv_p = 2*math.exp(-5*coop_angle / AA_MAX)-1
        else:
            adv_p = -1
    elif step_num <= 130:
        if antenna_train_angle < pi/6:
            adv_p = 2*math.exp(-6*antenna_train_angle / AA_MAX)-1
        else:
            adv_p = -1
    else:
        if aspect_angle < pi/5:
            adv_p = 2*math.exp(-5*aspect_angle / AA_MAX)-1
        else:
            adv_p = -1
    return adv_p
def velocity_adv(state, step_num):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b, z_m, coop_angle = state
    if step_num < 170:
        return -(v_r - v_b - 550) / 150.0
    else:
        return (v_r - v_b - 550) / 150.0

def _angle_adv(aa, ata):
    if aa <= AA_MAX:
        adv_rr = math.exp(-aa / AA_MAX)
    else:
        adv_rr = 0

    if ata > ATA_MAX:
        # adv_b = - math.exp(-(pi - antenna_train_angle) / ATA_MAX)
        adv_rb = 0
    else:
        adv_rb = math.exp(-ata / ATA_MAX)
    return adv_rr + adv_rb

def coop_angle_adv(state):
    distance, aspect_angle, antenna_train_angle, z_r, z_b, v_r, v_b, pitch_r, pitch_b, roll_r, roll_b, z_m, coop_angle = state

    adv_p = 2*math.exp(-np.abs(coop_angle-pi/2) / AA_MAX)-1

    return adv_p