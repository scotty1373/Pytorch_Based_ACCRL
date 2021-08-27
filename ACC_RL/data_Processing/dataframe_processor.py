# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import datetime as dt

# vaild sele
# MENU = sys.argv[1]
# FILE_PATH = sys.argv[2]

# test sele
MENU = 'graph'
FILE_PATH = 'train_log_1629914171.txt'
Timestamp = time.time()

date = str(dt.date.today)[1] + str(dt.date.today)[2]


# Internal document processing(singal line)
def str2list(_data_line):
    _data_line = _data_line.rstrip('\n').split(' ', 26)
    _data_line.pop(18)
    _frame_list = []
    _seq = [t for t in range(1, 26, 2)]
    for t in _seq:
        _frame_list.append(_data_line[t])
    return _frame_list


# Txt file to csv model
def txt2csv(path):
    fp = open(path, "r")
    # 临时变量
    data_lens_temp = fp.readlines()
    data_lens = len(data_lens_temp)
    print('file %s lines read: %s' %(path, str(data_lens)) )
    # 内存释放
    del data_lens_temp
    fp.seek(0, 0)
    # 参数初始化s
    df_list = np.zeros(shape=(data_lens, 13))
    for _i in range(data_lens):
        data_pre_line = fp.readline()
        df_list[_i] = str2list(data_pre_line)
    df_dataset = pd.DataFrame(df_list)
    df_dataset.columns = ['EPISODE', 'TIMESTAMP', 'EPISODE_LENGTH', 'ACTION',
                    'REWARD', 'Avg_REWARD', 'training_Loss', 'Q_MAX',
                    'gap', 'v_ego', 'v_lead', 'time', 'a_ego']
    df_dataset.to_csv(f'train_csv{str(Timestamp).split(".")[0]}.csv')
    print('txt transfom to csv successful')
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
    title = ('acc_info')
    plt.title(title, fontsize=20)
    # plt.grid(axis='y',color='grey',linestyle='--',lw=0.5,alpha=0.5)
    # plt.tick_params(axis='both',labelsize=14)
    plot1 = ax1.plot(epoch_iter, gap_mean, 'r')
    ax1.set_ylabel('gap', fontsize = 18)
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
    epoch = int(EPISODE[-1])
    epoch_iter = [i for i in range(epoch)]
    crash_index = np.argwhere(gap <= 3)
    lose_index = np.argwhere(gap >= 300)
    done_index = np.argwhere(EPISODE_LENGTH == 480)
    # crash/loss/done index 
    gap_crash = gap[crash_index[:, 0]]
    gap_loss = gap[lose_index[:, 0]]
    gap_done = gap[done_index[:, 0]]
    
    action_mean = []
    for i in epoch_iter:
        start_index = np.argwhere(EPISODE == i)[0][0]
        end_index = np.argwhere(EPISODE == i)[-1][0]
        action_mean.append(np.array(ACTION[start_index:end_index].mean()))
    fig, ax1 = plt.subplots(figsize=(10, 3))
    title = 'acc_info'
    print(f'Crash index:{EPISODE[crash_index[:, 0]].reshape(1, -1)}')
    print(f'Loss index:{EPISODE[lose_index[:, 0]].reshape(1,-1)}')
    plt.title(title, fontsize=20)
    # plt.grid(axis='y',color='grey',linestyle='--',lw=0.5,alpha=0.5)
    # plt.tick_params(axis='both',labelsize=14)
    plot1 = ax1.scatter(EPISODE[crash_index[:, 0].reshape(len(gap_crash), 1)], gap_crash, c='red')
    plot2 = ax1.scatter(EPISODE[lose_index[:, 0].reshape(len(gap_loss), 1)], gap_loss, c='blue')
    plot3 = ax1.scatter(EPISODE[done_index[:, 0].reshape(len(gap_done), 1)], gap_done, c='green')
    ax1.set_ylabel('gap', fontsize=18)
    ax2 = ax1.twinx()
    plot3 = ax2.plot(epoch_iter, action_mean, 'g')
    plt.show()
    ax2.set_ylabel('action', fontsize=18)
    return crash_index, lose_index

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
    plt.plot(time_stamp, Qmax)
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
                                                                         ACTION_, REWARD_, index_-1)
    # Plot val in graph
    plt.subplot(411)
    v_lead_g, = plt.plot(length_ep, v_lead_, linewidth=2, color='C1')
    v_ego_g, = plt.plot(length_ep, v_ego_, linewidth=2, color='C9')
    gap_g, = plt.plot(length_ep, gap_, linewidth=2, color='C3', linestyle=':')
    
    plt.legend(handles=[v_lead_g, v_ego_g, gap_g],
               labels=['v_lead', 'v_ego', 'gap'], loc=2)
    plt.title('info_{}'.format(index_-1))
    plt.xlabel('Epoch', size=10)
    plt.ylabel('info_{}'.format(index_-1), size=10)
    
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
    # Plot val in graph
    plt.subplot(413)
    v_lead_g, = plt.plot(length_ep, v_lead_, linewidth=2, color='C1')
    v_ego_g, = plt.plot(length_ep, v_ego_, linewidth=2, color='C9')
    gap_g, = plt.plot(length_ep, gap_, linewidth=2, color='C3', linestyle=':')
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
    

if __name__ == '__main__':
    if MENU == 'proc':
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
        time = np.array(df.iloc[0:row, 12:13], dtype=float)
        a_ego = np.array(df.iloc[0:row, 13:14], dtype=float)
        # plot graph
        plot_action_reward_gap_v_(EPISODE, ACTION, gap, v_ego, v_lead)
        plot_Qmax_singel_timeframe(Q_MAX, TIMESTAMP)
        indexOfcrash, indexoflose = plot_reward_action_crash(EPISODE, ACTION, gap, EPISODE_LENGTH)
        while True:
            crash_index = input('Enter the crash index you want to view: ')
            plot_singal_info(EPISODE, EPISODE_LENGTH, v_lead, v_ego, gap, ACTION, REWARD, int(crash_index))


