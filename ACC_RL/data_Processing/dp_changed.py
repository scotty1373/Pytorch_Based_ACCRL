# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import datetime as dt
from numpy import seterr

seterr(all='raise')
# np.set_printoptions(suppress=True, threshold=10000, linewidth=10000)
np.set_printoptions(suppress=True, threshold=10000)

# vaild sele
# MENU = sys.argv[1]
# FILE_PATH = sys.argv[2]

# test sele
MENU = 'proc'
FILE_PATH = 'train_log_1636640661_yolo面积计算.txt'
Timestamp = time.time()

date = str(dt.date.today)[1] + str(dt.date.today)[2]
data_num_in_line = 24
episode_filter = 349


# Internal document processing(single line)
def str2list(_data_line, version_old=False):
    _frame_list = []
    #               已更新训练代码
    # 由于gap与v_ego数据之间有空行，需要抛去空行所占strip位置
    if version_old:
        _data_line = _data_line.rstrip('\n').split(' ', 26)
        _data_line.pop(18)
        _seq = [t for t in range(1, 26, 2)]
    else:
        _data_line = _data_line.rstrip('\n').split(' ', data_num_in_line)
        _seq = [t for t in range(1, data_num_in_line, 2)]
    for t in _seq:
        _frame_list.append(_data_line[t])
    return _frame_list


# Txt file to csv model
def txt2csv(path, old=False):
    fp = open(path, "r")
    # 临时变量
    data_lens_temp = fp.readlines()
    data_lens = len(data_lens_temp)
    print('file %s lines read: %s' % (path, str(data_lens)))
    # 内存释放
    del data_lens_temp
    fp.seek(0, 0)
    # 参数初始化s
    if old:
        df_list = np.zeros(shape=(data_lens, int(26 / 2)))
    else:
        df_list = np.zeros(shape=(data_lens, int(data_num_in_line / 2)))
    for _i in range(data_lens):
        data_pre_line = fp.readline()
        df_list[_i] = str2list(data_pre_line, old)
    df_dataset = pd.DataFrame(df_list)

    if old:
        df_dataset.columns = ['EPISODE', 'TIMESTAMP', 'EPISODE_LENGTH', 'ACTION',
                              'REWARD', 'Avg_REWARD', 'train_loss', 'Qmax',
                              'gap', 'v_ego', 'v_lead', 'time', 'a_ego']
    else:
        df_dataset.columns = ['EPISODE', 'TIMESTAMP', 'EPISODE_LENGTH', 'ACTION',
                              'REWARD', 'Avg_REWARD', 'train_loss', 'Qmax',
                              'gap', 'v_ego', 'v_lead', 'a_ego']
    df_dataset.to_csv(f'{os.path.splitext(path)[0]}.csv')
    print('txt transform to csv successful')
    fp.close()


# Plot figure with index ACTION + REWARD + GAP + SPEED
def plot_action_reward_gap_v_(EPISODE, ACTION, gap, v_ego, v_lead):
    epoch = int(EPISODE[-1])
    epoch_iter = [i for i in range(epoch)]
    gap_mean = []
    action_mean = []
    for i in epoch_iter:
        start_index = np.argwhere(EPISODE == i)[0][0]
        end_index = np.argwhere(EPISODE == i)[-1][0]
        # 对每个epoch取对应长度数据的平均
        gap_mean.append(np.array(gap[start_index:end_index].mean()))
        action_mean.append(np.array(ACTION[start_index:end_index].mean()))
    fig, ax1 = plt.subplots(figsize=(10, 3))
    title = 'acc_info'
    plt.title(title, fontsize=20)
    # plt.grid(axis='y',color='grey',linestyle='--',lw=0.5,alpha=0.5)
    # plt.tick_params(axis='both',labelsize=14)
    plot1 = ax1.plot(epoch_iter, gap_mean, 'r')
    ax1.set_ylabel('gap', fontsize=18)
    ax2 = ax1.twinx()
    plot2 = ax2.plot(epoch_iter, action_mean, 'g')
    plt.show()
    ax2.set_ylabel('action', fontsize=18)

    # ax2.tick_params(axis='y',labelsize=14)
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('g')                    
    # ax2.set_xlim(1966,2014.15)
    # lines = plot1 + plot2           
    # ax1.legend(lines,[l.get_label() for l in lines])                       
    # plt.savefig("train_test{ }.png".format(date))


# Plot with CARSH and LOSS time
def plot_reward_action_crash(EPISODE, ACTION, gap, EPISODE_LENGTH):
    epoch = episode_filter
    filter_index = np.argwhere(EPISODE == episode_filter)[-1][0]
    epoch_iter = [i for i in range(epoch)]
    crash_index_ = np.argwhere(gap[:filter_index, :] <= 3)
    loss_index_ = np.argwhere(gap[:filter_index, :] >= 300)
    done_index_ = np.argwhere(EPISODE_LENGTH[:filter_index, :] == 480)
    # crash/loss/done index 
    gap_crash = gap[crash_index_[:, 0]]
    gap_loss = gap[loss_index_[:, 0]]
    gap_done = gap[done_index_[:, 0]]

    action_mean = []
    for i in epoch_iter:
        start_index = np.argwhere(EPISODE == i)[0][0]
        end_index = np.argwhere(EPISODE == i)[-1][0]
        action_mean.append(np.array(ACTION[start_index:end_index].mean()))
    fig, ax1 = plt.subplots(figsize=(10, 3))
    title = 'policy_based'
    print(f'Crash index:{EPISODE[crash_index_[:, 0]].reshape(1, -1)}')
    print(f'Loss index:{EPISODE[loss_index_[:, 0]].reshape(1, -1)}')
    plt.title(title, fontsize=10)
    # plt.grid(axis='y',color='grey',linestyle='--',lw=0.5,alpha=0.5)
    # plt.tick_params(axis='both',labelsize=14)
    index_for_x_ticks = np.array([150, 300])
    index_for_y_ticks = np.array([0, 50, 300])

    plot1 = ax1.scatter(EPISODE[crash_index_[:, 0].reshape(len(gap_crash), 1)], gap_crash, c='red')
    plot2 = ax1.scatter(EPISODE[loss_index_[:, 0].reshape(len(gap_loss), 1)], gap_loss, c='blue')
    plot3 = ax1.scatter(EPISODE[done_index_[:, 0].reshape(len(gap_done), 1)], gap_done, c='green')
    plt.legend((plot1, plot2, plot3), ('crash', 'loss_track', 'complete'), loc='best')
    plt.xticks(index_for_x_ticks)
    plt.yticks(index_for_y_ticks)
    plt.xlabel('Training Episode')
    ax1.set_ylabel('Gap', fontsize=10)
    plt.tick_params(labelsize=10)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    # ax2 = ax1.twinx()
    # plot3 = ax2.plot(epoch_iter, action_mean, 'g')
    plt.savefig('data', dpi=300)
    plt.show()
    # ax2.set_ylabel('action', fontsize=18)
    return crash_index_, loss_index_

    # plt.figure(figsize=(8, 5))
    # action, = plt.plot(epoch_iter, action_mean, linewidth=2, color='red')
    # gap_, = plt.plot(epoch_iter, np.array(gap_mean), linewidth=2, color='blue')
    # v_ego_, = plt.plot(EPISODE, v_ego, linewidth=2, color='yellow')
    # v_lead_, = plt.plot(EPISODE, v_lead, linewidth=2, color='k')
    # plt.legend(handles=[action, gap_, v_ego_, v_lead_], labels=['ACTION', 'gap', 'v_ego', 'v_lead'], loc='best')
    # plt.title('acc_info')
    # plt.xlabel('Epoch', size=10)
    # plt.ylabel('info', size=10)
    # plt.show()


def plot_Qmax_singel_timeframe(Qmax, time_stamp):
    plt.title('Qmax generator')
    plt.plot(time_stamp[0::10], Qmax[0::10])
    plt.show()


def get_singal_info(EPISODE, EPISODE_LENGTH, v_lead, v_ego, gap, ACTION, REWARD, index):
    epoch = int(EPISODE[-1])
    epoch_iter = [i for i in range(epoch)]
    start_idx = np.argwhere(EPISODE == index)[0][0]
    end_idx = np.argwhere(EPISODE == index)[-1][0]
    length_ep = [t for t in range(1, end_idx - start_idx + 1)]
    # 参数范围划定
    v_lead_ = v_lead[start_idx: end_idx]
    v_ego_ = v_ego[start_idx: end_idx]
    gap_ = gap[start_idx: end_idx]
    action_ = ACTION[start_idx: end_idx]
    reward_ = REWARD[start_idx: end_idx]
    return length_ep, v_lead_, v_ego_, gap_, action_, reward_


def plot_singal_info(EPISODE_, EPISODE_LENGTH_, _v_lead, _v_ego, _gap, ACTION_, REWARD_, index_):
    length_ep, v_lead_, v_ego_, gap_, action_, reward_ = get_singal_info(EPISODE_, EPISODE_LENGTH_,
                                                                         _v_lead, _v_ego, _gap,
                                                                         ACTION_, REWARD_, index_ - 1)
    # Plot val in graph
    ax_a_g = plt.subplot(411)
    ax_a_v = ax_a_g.twinx()
    v_lead_g, = ax_a_v.plot(length_ep, v_lead_, linewidth=2, color='C1')
    v_ego_g, = ax_a_v.plot(length_ep, v_ego_, linewidth=2, color='C9')
    gap_g, = ax_a_g.plot(length_ep, gap_, linewidth=2, color='C3', linestyle=':')

    plt.legend(handles=[v_lead_g, v_ego_g, gap_g],
               labels=['v_lead', 'v_ego', 'gap'], loc=2)
    plt.title('info_{}'.format(index_ - 1))
    plt.xlabel('Epoch', size=10)
    plt.ylabel('info_{}'.format(index_ - 1), size=10)

    plt.subplot(412)
    action_g, = plt.plot(length_ep, action_, linewidth=2, color='C4')
    reward_g, = plt.plot(length_ep, reward_, linewidth=2, color='C5', linestyle=':')
    plt.legend(handles=[action_g, reward_g],
               labels=['action', 'reward'], loc=2)
    # plt.title('info_{}'.format(_index-1))
    # plt.xlabel('Epoch', size=10)
    # plt.ylabel('info_{}'.format(_index-1), size=10)

    length_ep, v_lead_, v_ego_, gap_, action_, reward_ = get_singal_info(EPISODE_,
                                                                         EPISODE_LENGTH_, _v_lead, _v_ego,
                                                                         _gap, ACTION_, REWARD_, index_)
    '''
    # 制作数据，处理数据 >>>
    v_relative = (v_ego_[:, 0] - v_lead_[:, 0]).reshape(-1, 1)
    acc_relative = np.zeros((len(length_ep), 1))
    for index in range(1, len(length_ep)):
        try:
            acc_relative[index, :] = (v_relative[index, :]**2 - v_relative[index - 1, :]**2) / (2 * (gap_[index, :] - gap_[index-1, :]))
        except FloatingPointError as e:     # numpy将所有0/0错误归于FloatingPointError，第十行定义numpy抛出所有警告类型为错误，即可捕获RunTimeWarning
            print(f"index {index} gap has no change")
            acc_relative[index, :] = 0
    print(np.concatenate([np.arange(0, len(length_ep)).reshape(-1, 1), v_lead_, v_ego_, gap_, action_, reward_, v_relative, acc_relative], axis=1))

    # 制作数据，处理数据 <<<
    '''

    # Plot val in graph
    ax_b_g = plt.subplot(413)
    ax_b_v = ax_b_g.twinx()
    v_lead_g, = ax_b_v.plot(length_ep, v_lead_, linewidth=2, color='C1')
    v_ego_g, = ax_b_v.plot(length_ep, v_ego_, linewidth=2, color='C9')
    gap_g, = ax_b_g.plot(length_ep, gap_, linewidth=2, color='C3', linestyle=':')
    plt.legend(handles=[v_lead_g, v_ego_g, gap_g],
               labels=['v_lead', 'v_ego', 'gap'], loc=2)
    plt.title('info_{}'.format(index_))
    plt.xlabel('Epoch', size=10)
    plt.ylabel('info_{}'.format(index_), size=10)

    plt.subplot(414)
    action_g, = plt.plot(length_ep, action_, linewidth=2, color='C4')
    reward_g, = plt.plot(length_ep, reward_, linewidth=2, color='C5', linestyle=':')
    plt.legend(handles=[action_g, reward_g],
               labels=['action', 'reward'], loc=2)

    plt.show()


def relative(EPISODE_, EPISODE_LENGTH_, _v_lead, _v_ego, _gap, ACTION_, REWARD_, index_):
    length_ep, v_lead_, v_ego_, gap_, action_, reward_ = get_singal_info(EPISODE_, EPISODE_LENGTH_,
                                                                         _v_lead, _v_ego, _gap,
                                                                         ACTION_, REWARD_, index_)

    acc_lead_ = np.zeros(((len(length_ep)), 1))
    acc_ego_ = np.zeros(((len(length_ep)), 1))

    acc_lead_[1:, :] = (v_lead_[1:, :] - v_lead_[:-1, :]) / 0.5
    acc_ego_[1:, :] = (v_ego_[1:, :] - v_ego_[:-1, :]) / 0.5
    acc_compare = acc_ego_ - acc_lead_

    # 制作数据，处理数据 >>>
    v_relative = (v_ego_[:, 0] - v_lead_[:, 0]).reshape(-1, 1)
    gap_relative = np.zeros((len(length_ep), 1))
    gap_relative[1:, :] = (gap_[1:, 0] - gap_[:-1, 0]).reshape(-1, 1)
    acc_relative = np.zeros((len(length_ep), 1))
    for index in range(1, len(length_ep)):
        try:
            acc_relative[index, :] = (v_relative[index, :] ** 2 - v_relative[index - 1, :] ** 2) / (
                        2 * (gap_[index, :] - gap_[index - 1, :]))
        except FloatingPointError as e:  # numpy将所有0/0错误归于FloatingPointError，第十行定义numpy抛出所有警告类型为错误，即可捕获RunTimeWarning
            print(f"index {index} gap has no change")
            acc_relative[index, :] = 0

    # reward_recal = Caculate_reward(v_relative, gap_, acc_relative)
    try:
        ttc = gap_ / v_relative
    except FloatingPointError as e:
        pass

    '''
    reward_gap = (np.exp(-(gap_ - 50)**2 / (2 * 5.3**2)) / (np.sqrt(2*np.pi) * 5.3)) * 100 / 7.9

    reward_recal = np.zeros((len(length_ep), 1))
    for index in range(len(length_ep)):
        if ttc[index, :] < 0:
            reward_recal[index, :] = (reward_gap[index, :] - 0.2) / 0.75
        elif ttc[index, :] >= 0:
            reward_recal[index, :] = scti_Caculate(ttc[index, :]) * 0.5 + (reward_gap[index, :] - 0.2) * 0.5 / 0.75

    print(np.concatenate([np.arange(0, len(length_ep)).reshape(-1, 1), v_lead_, v_ego_, gap_, action_, reward_, v_relative, acc_relative, acc_compare, (reward_gap - 0.2)/0.75, reward_recal], axis=1))
    '''

    # 计算距离确定公式是否正确
    t = 3.5
    distance_ef = gap_ - (v_relative * 0.5 + 0.5 * acc_compare * 0.5 ** 2)
    action_best = (-(50 - gap_) - v_relative * t) / (2 * t ** 2)

    reward_recal = np.zeros((len(length_ep), 1))
    for index in range(len(length_ep)):
        # reward_recal[index, :] = np.exp((action_[index, :] - 0.5 - (-(50 - gap_[index, :] - v_relative[index, :] * t) / ((2 * t) ** 2))) / 2 * 0.5 ** 2) / (np.sqrt(2 * np.pi) * 0.5) * 0.5 / 0.8
        try:
            reward_recal[index, :] = (np.exp(
                -(acc_compare[index, :] - action_best[index, :]) ** 2 / (2 * (0.3 ** 2))) / (
                                                  np.sqrt(2 * np.pi) * 0.3)) / 1.4
        except FloatingPointError as e:
            reward_recal[index, :] = 0
    print(np.concatenate(
        [np.arange(0, len(length_ep)).reshape(-1, 1), v_lead_, v_ego_, gap_, action_, reward_, action_best, acc_compare,
         acc_ego_, acc_lead_, reward_recal], axis=1))
    print(f'acc max:{acc_compare[:, 0].max()}')
    print(f'acc min:{acc_compare[:, 0].min()}')
    print(f'v_relative max:{v_relative[:, 0].max()}')
    print(f'v_relative min:{v_relative[:, 0].min()}')

    '''
    # 制作数据，处理数据 <<<
    fig, axis = plt.subplots()
    axis2 = axis.twinx()
    v_rel, = axis2.plot(length_ep, v_relative, linewidth=2, color='C1')
    gap_rel, = axis.plot(length_ep, gap_relative, linestyle='-', color='C3')
    acc_rel, = axis2.plot(length_ep, acc_compare, linewidth=2, color='C9')
    # gap_g, = ax_b_g.plot(length_ep, gap_, linewidth=2, color='C3', linestyle=':')
    plt.legend(handles=[v_rel, gap_rel, acc_rel],
               labels=['v_rel', 'gap_rel', 'acc_rel'], loc='best')
    plt.show()
    '''


def scti_Caculate(ttc_min, ttc_=8):
    if ttc_min <= ttc_:
        scti = (100 * np.power(ttc_min, 1.4)) / (np.power(ttc_min, 1.4) + np.power(ttc_ - ttc_min, 1.5)) / 100
    elif ttc_min - ttc_ > 100:
        scti = -1
    else:
        scti = (100 * np.exp((-np.power((ttc_min - ttc_), 2)) / (2 * np.power(ttc_, 2)))) / 100
    return scti


def Caculate_reward(v_ref, g_ref, a_ref):
    a1 = -1.08e-3
    a2 = 1.136e-4
    a3 = -1.643e-2
    a4 = 9.927e-4
    a5 = -002.163e-3
    a6 = -1.6435e-2
    a7 = -7.3387e-2
    a8 = -2.0589e-2
    a9 = 6.83969e-2
    a10 = 1.116254
    Function = a1 * (v_ref ** 2) + a2 * (g_ref ** 2) + a3 * (
                a_ref ** 2) + a4 * v_ref * g_ref + a5 * g_ref * a_ref + a6 * v_ref * a_ref + a7 * v_ref + a8 * g_ref + a9 * a_ref + a10

    return Function


def data_concat(*args):
    data = np.concatenate(*args, axis=1)
    data_clean = np.where(data[:, 3] < 0.3)


def prof_reward_adder(ep_, ep_len_, reward_):
    epoch_iter = int(ep_[-1])
    reward_epoch_adder = np.zeros((1, epoch_iter))
    ep_len_adder = np.zeros((1, epoch_iter))
    for index in range(epoch_iter):
        start_idx = np.argwhere(ep_ == index)[0][0]
        end_idx = np.argwhere(ep_ == index)[-1][0]
        length_ep = [t for t in range(1, end_idx - start_idx + 1)]
        reward_epoch_adder[:, index] = np.array(reward_[start_idx:end_idx, :]).sum()
        ep_len_adder[:, index] = np.array(length_ep[-1])
    plt.plot(np.arange(0, epoch_iter), reward_epoch_adder.flatten())
    plt.show()


if __name__ == '__main__':
    if MENU == 'proc':
        args_file = input('The log file record is old version(y/n): ')
        if args_file == 'y':
            txt2csv(FILE_PATH, True)
        else:
            txt2csv(FILE_PATH)
    elif MENU == 'graph':
        list_dir = os.listdir()
        print('File under current path: ', list_dir)
        CSV_FILE_NAME = input('choose your CSV file: ')
        df = pd.read_csv(CSV_FILE_NAME)
        row, col = df.shape
        # 参数读取
        INDEX = np.array(range(row))
        EPISODE = np.array(df.iloc[0:row, 1:2], dtype=int)
        TIMESTAMP = np.array(df.iloc[0:row, 2:3], dtype=float)
        EPISODE_LENGTH = np.array(df.iloc[0:row, 3:4], dtype=int)
        ACTION = np.array(df.iloc[0:row, 4:5], dtype=float)
        REWARD = np.array(df.iloc[0:row, 5:6], dtype=float)
        Avg_REWARD = np.array(df.iloc[0:row, 6:7], dtype=float)
        training_Loss = np.array(df.iloc[0:row, 7:8], dtype=float)
        Q_MAX = np.array(df.iloc[0:row, 8:9], dtype=float)
        gap = np.array(df.iloc[0:row, 9:10], dtype=float)
        v_ego = np.array(df.iloc[0:row, 10:11], dtype=float)
        v_lead = np.array(df.iloc[0:row, 11:12], dtype=float)

        prof_reward_adder(EPISODE, EPISODE_LENGTH, REWARD)

        # plot graph
        plot_action_reward_gap_v_(EPISODE, ACTION, gap, v_ego, v_lead)
        plot_Qmax_singel_timeframe(Q_MAX, TIMESTAMP)
        index_of_crash, index_of_loss = plot_reward_action_crash(EPISODE, ACTION, gap, EPISODE_LENGTH)
        while True:
            crash_index = input('Enter the crash index you want to view: ')
            relative(EPISODE, EPISODE_LENGTH, v_lead, v_ego, gap, ACTION, REWARD, int(crash_index))
            plot_singal_info(EPISODE, EPISODE_LENGTH, v_lead, v_ego, gap, ACTION, REWARD, int(crash_index))
