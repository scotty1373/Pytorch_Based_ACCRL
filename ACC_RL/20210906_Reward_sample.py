import base64
import copy
import datetime as dt
import io
import os
import random
import socket
import sys
import time
from collections import deque
from utils_tools.net_builder import Data_dim_reduce as build_model
import numpy as np
import skimage
import torch
from torch.autograd import Variable

from PIL import Image, ImageDraw
from skimage import color

np.set_printoptions(precision=4)

EPISODES = 350
img_rows, img_cols = 80, 80
Distance_EF = 50
Return_Time = 7
Variance = 0.5
# Convert image into gray scale
# We stack 8 frames, 0.06*8 sec
img_channels = 4 
unity_Block_size = 65536
# PATH_MODEL = 'C:/dl_data/Python_Project/save_model/'
# PATH_LOG = 'C:/dl_data/Python_Project/train_log/'
# C:\DRL_data\Python_Project\Pytorch_Learning-multi-Thread\ACC_RL\save_Model\save_model_1633081266
CHECK_POINT_TRAIN_PATH = './save_Model/save_model_1633081266/save_model_298.h5'
PATH_MODEL = 'save_Model'
PATH_LOG = 'train_Log'
time_Feature = round(time.time())
random_index = np.random.permutation(img_channels)


class DQNAgent:
    def __init__(self, state_size_, action_size_, device_):
        self.t = 0
        self.max_Q = 0
        self.trainingLoss = 0
        self.train = True
        self.train_from_checkpoint = False

        # Get size of state and action
        self.state_size = state_size_
        self.action_size = action_size_
        self.device = device_

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if self.train and not self.train_from_checkpoint:
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 0
            self.initial_epsilon = 0
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 100
        self.train_from_checkpoint_start = 3000
        self.explore = 4000

        # Create replay memory using deque
        self.memory = deque(maxlen=32000)

        # Create main model and target model
        self.model = build_model().to(self.device)
        self.target_model = build_model().to(self.device)

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss = torch.nn.MSELoss()

    def process_image(self, obs):
        obs = skimage.color.rgb2gray(obs)
        return obs
        # camera_info = CamInfo({
        #     "f_x": 500/5*8,         # focal length x
        #     "f_y": 500/5*8,         # focal length y
        #     "u_x": 200,             # optical center x
        #     "u_y": 200,             # optical center y
        #     "camera_height": 1400,  # camera height in `mm`
        #     "pitch": 90,            # rotation degree around x
        #     "yaw": 0                # rotation degree around y
        # })
        # ipm_info = CamInfo({
        #     "input_width": 400,
        #     "input_height": 400,
        #     "out_width": 80,
        #     "out_height": 80,
        #     "left": 0,
        #     "right": 400,
        #     "top": 200,
        #     "bottom": 400
        # })
        # ipm_img = IPM(camera_info, ipm_info)
        # out_img = ipm_img(obs)
        # if gap < 10:
        #     skimage.io.imsave('outimage_' + str(gap) + '.png', out_img)

        # return out_img

    def update_target_model(self):
        # 解决state_dict浅拷贝问题
        weight_model = copy.deepcopy(self.model.state_dict())
        self.target_model.load_state_dict(weight_model)

    # Get action from model using epsilon-greedy policy
    def get_action(self, Input):
        if np.random.rand() <= self.epsilon:
            # print("Return Random Value")
            # return random.randrange(self.action_size)
            return np.random.uniform(-1, 1)
        else:
            # print("Return Max Q Prediction")
            q_value = self.model(Input[0], Input[1])
            # Convert q array to steering value
            return linear_unbin(q_value[0])

    def replay_memory(self, state, v_ego, action, reward, next_state, nextV_ego, done):
        self.memory.append((state, v_ego, action, reward, next_state, nextV_ego, done, self.t))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    # @profile
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        elif self.train_from_checkpoint:
            if len(self.memory) < self.train_from_checkpoint_start:
                return
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        '''
        torch.float64对应torch.DoubleTensor
        torch.float32对应torch.FloatTensor
        '''
        state_t, v_ego_t, action_t, reward_t, state_t1, v_ego_t1, terminal, step = zip(*minibatch)
        state_t = Variable(torch.Tensor(state_t).squeeze().to(self.device))
        state_t1 = Variable(torch.Tensor(state_t1).squeeze().to(self.device))
        v_ego_t = Variable(torch.Tensor(v_ego_t).squeeze().to(self.device))
        v_ego_t1 = Variable(torch.Tensor(v_ego_t1).squeeze().to(device))

        self.optimizer.zero_grad()

        targets = self.model(state_t, v_ego_t)
        self.max_Q = torch.max(targets[0]).item()
        target_val = self.model(state_t1, v_ego_t1)
        target_val_ = self.target_model(state_t1, v_ego_t1)
        for i in range(batch_size):
            if terminal[i] == 1:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = torch.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])
        logits = self.model(state_t, v_ego_t)
        loss = self.loss(logits, targets)
        loss.backward()
        self.optimizer.step()
        self.trainingLoss = loss.item()

    def load_model(self, name):
        checkpoints = torch.load(name)
        self.model.load_state_dict(checkpoints['model'])
        self.optimizer.load_state_dict(checkpoints['optimizer'])

    # Save the model which is under training
    def save_model(self, name):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, name)


# 单目标斜对角坐标
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyxy2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2]
    y[:, 3] = x[:, 3]
    y = y.type(torch.IntTensor)
    return y


def linear_bin(a):
    """
    Convert a value to a categorical array.
    Parameters
    ----------
    a : int or float
        A value between -1 and 1
    Returns
    -------
    list of int
        A list of length 21 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 20))
    arr = np.zeros(21)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.
    See Also
    --------
    linear_bin
    """
    arr = arr.data.cpu().numpy()
    if not len(arr) == 21:
        raise ValueError('Illegal array length, must be 21')
    b = np.argmax(arr)
    a = b * 2 / 20 - 1
    return a
# def oberve():
#   revcData, (remoteHost, remotePort) = sock.recvfrom(65536)


def decode(revcData, v_ego_=0, v_lead_=0, force=0, episode_len=0, v_ego_copy_=0, v_lead_copy_=0):
    # received data processing
    revcList = str(revcData).split(',', 4)
    gap_ = float(revcList[0][2:])                    # distance between vehicles
    v_ego1_ = float(revcList[1])                      # speed of egoVehicle
    v_lead1_ = float(revcList[2])                     # speed of leadVehicle
    a_ego1_ = float(revcList[3])                      # acceleration of egoVehicle
    img = base64.b64decode(revcList[4])      # image from mainCamera
    image = Image.open(io.BytesIO(img))
    # image resize, 双线性插值
    images = np.array(image)

    anchor_box = yolo(images)
    # !!!后期可能会出现问题（问题具体出现在锚框不准确的问题）!!!
    # 后期优化方向，取置信度最大的锚框index作为锚框选取坐标
    # index = anchor_box.pred[0][0, 4]

    # 当前代码问题，yolo输出锚框为1时选取锚框问题，暂时解决通过取消squeeze
    # !!!锚框缺失问题!!!
    position = np.array(xyxy2xyxy(anchor_box.pred[0]), dtype='uint8')

    # pillow 支持
    if position.shape[0] != 0:
        ImageDraw.Draw(image).rectangle([(position[0, 0], position[0, 1]), (position[0, 2], position[0, 3])], outline='yellow', width=3)
    image = image.resize((80, 80), resample=Image.BILINEAR)
    image = np.array(image)

    '''
    opencv支持
    if position.shape[0] != 0:
        image = cv2.rectangle(image, (position[0, 0], position[0, 1]), (position[0, 2], position[0, 3]), (0, 255, 0), 2)
    # 更改双线性插值为区域插值，图像处理完效果好于双线性插值
    image = cv2.resize(image, (80, 80), interpolation=cv2.INTER_AREA)
    '''

    # 计算reward
    done_ = 0
    v_relative = v_ego1_ - v_lead1_
    action_relative = ((v_ego1_ - v_ego_) / 0.5) - ((v_lead1_ - v_lead_) / 0.5)
    action_best = (2 * (-(Distance_EF - float(gap_)) - v_relative * Return_Time)) / (Return_Time ** 2)

    # 动力学公式
    # reward_ = CalReward(action_relative, action_best, gap_)
    # 平滑reward
    reward_ = cal_reward(gap_, v_ego1_, force, episode_len)

    if gap_ <= 3 or gap_ >= 300:
        done_ = 1
        reward_ = -1.0
    elif episode_len > 480:
        done_ = 2
        # reward = CalReward(float(gap), float(v_ego), float(v_lead), force)
    return image, float(reward_), done_, float(gap_), float(v_ego1_), float(v_lead1_), float(a_ego1_)


# 修改正态分布中的方差值，需要重新将正态分布归一化
def CalReward(action_relative_, action_best_, gap_):
    try:
        reward_recal = (np.exp(-(action_relative_ - action_best_) ** 2 / (2 * (Variance ** 2))))
    except FloatingPointError as e:
        reward_recal = 0

    if 40 <= gap_ <= 60:
        pass
    elif 5 < gap_ < 40:
        reward_recal = reward_recal * ((1 / 35) * gap_ - (1 / 7))
    elif 60 <= gap_ <= 300:
        reward_recal = reward_recal * (- (1 / 240) * gap_ + (5 / 4))
    return reward_recal


def cal_reward(gap_, x, y, ep_len):
    p00 = 1.007
    p10 = -0.01203
    p01 = 1.516e-06
    p20 = -0.0002435
    p11 = -4.13e-07
    p02 = -0.04686
    p30 = 1.698e-05
    p21 = 1.064e-08
    p12 = -0.03628
    p03 = 3.062e-06
    Rp = p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*x**2*y + p12*x*y**2 + p03*y**3

    if x > 30:
        Rp = 0
    # reward for gap
    if 50 <= gap_ <= 70:
        Rd = 1
    elif 30 <= gap_ < 50:
        Rd = 0.5
    elif 70 < gap_ <= 100:
        Rd = 0.5
    else:
        Rd = 0.0

    # reward for gap poly
    # r1 = -2.428e-03
    # r2 = -2.340e-05
    # r3 = 7.463e-06
    # r4 = -4.440e-01
    # r5 = 5.132e-03
    # r6 = -2.674e-05
    # r7 = 5.246e-08
    # r8 = 14.82323484
    # r9 = 1e-03
    #
    # if 50 <= gap_ <= 70:
    #     Rd = 1
    # elif 3 <= gap_ < 50:
    #     Rd = r1*gap_ - r2*gap_**2 + r3*gap_**3
    # elif 70 < gap_ <= 150:
    #     Rd = r4*gap_ + r5*(gap_**2) + r6*(gap_**3) + r7*(gap_**4) + r8
    # else:
    #     Rd = 0.0
    # Rd = Rd + (r9 * ep_len)
    return Rp * Rd


def reset():
    strr = str(3) + ',' + '0.0'
    sendDataLen = sock.sendto(strr.encode(), (remoteHost, remotePort))


def print_out(file, text):
    file.write(text + '\n')
    file.flush()
    sys.stdout.flush()


# @profile
def thread_Train_init():
    global agent
    step_epsode = 0
    while True:
        if len(agent.memory) < agent.train_start:
            time.sleep(5)
            continue
        agent.train_replay()
        time.sleep(0.1)
        step_epsode += 1
        # print('train complete in num: %s' %str(step_epsode))


def log_File_path(path):
    # date = str(dt.date.today()).split('-')
    # date_concat = date[1] + date[2]
    date_concat = time_Feature
    train_log_ = open(os.path.join(path, 'train_log_{}.txt'.format(date_concat)), 'w')
    del date_concat
    return train_log_


def random_sample(state_t, v_t, state_t1, v_t1):
    # random_index = np.random.permutation(img_channels)
    state_t = state_t[:, :, :, random_index]
    v_t = v_t[:, random_index]
    state_t1 = state_t1[:, :, :, random_index]
    v_t1 = v_t1[:, random_index]
    return state_t, v_t, state_t1, v_t1


def Recv_data_Format(byte_size, _done, v_ego_=None, v_lead1_=None, action_=None, episode_len_=None, s_t_=None, v_ego_t_=None):
    if _done != 0: 
        revcData, (remoteHost_, remotePort_) = sock.recvfrom(byte_size)
        image, _, _, gap_, v_ego_, v_lead1_, a_ego = decode(revcData)
        x_t = agent.process_image(image)

        s_t_ = np.stack((x_t, x_t, x_t, x_t), axis=0)
        v_ego_t_ = np.array((v_ego_, v_ego_, v_ego_, v_ego_))

        # In Keras, need to reshape
        s_t_ = s_t_.reshape(1, s_t_.shape[0], s_t_.shape[1], s_t_.shape[2])     # 1*80*80*4
        v_ego_t_ = v_ego_t_.reshape(1, v_ego_t_.shape[0])   # 1*4
        return s_t_, v_ego_t_, v_ego_, v_lead1_, remoteHost_, remotePort_
    else:
        revcData, (remoteHost_, remotePort_) = sock.recvfrom(byte_size)

        image, reward_, done_, gap_, v_ego1_, v_lead1_, a_ego1_ = decode(revcData, v_ego_, v_lead1_, action_, episode_len_)

        x_t1 = agent.process_image(image)
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])     # 1x1x80x80
        s_t1_ = np.append(x_t1, s_t_[:, :3, :, :], axis=1)    # 1x4x80x80
        v_ego_1_reshape = np.array(v_ego1_)
        v_ego_1_reshape = np.expand_dims(v_ego_1_reshape, -1)
        v_ego_1_reshape = np.expand_dims(v_ego_1_reshape, -1)
        v_ego_t1_ = np.append(v_ego_1_reshape, v_ego_t_[:, :3], axis=1)   # 1x4
        return reward_, done_, gap_, v_ego1_, v_lead1_, a_ego1_, v_ego_1_reshape, s_t1_, v_ego_t1_


def Send_data_Format(remoteHost_, remotePort_, s_t_, v_ego_t_, episode_len_, UnityReset_):
    pred_time_pre = dt.datetime.now()
    episode_len_ = episode_len_ + 1
    # Get action for the current state and go one step in environment
    s_t_ = torch.Tensor(s_t_).to(device)
    v_ego_t_ = torch.Tensor(v_ego_t_).to(device)
    force = agent.get_action([s_t_, v_ego_t_])
    action = force
      
    if UnityReset_ == 1:
        strr = str(4) + ',' + str(action)
        UnityReset_ = 0
    else:
        strr = str(1) + ',' + str(action)
    
    sendDataLen = sock.sendto(strr.encode(), (remoteHost_, remotePort_))    # 0.06s later receive
    pred_time_end = dt.datetime.now()
    time_cost = pred_time_end - pred_time_pre
    return episode_len_, action, time_cost, UnityReset_


def Model_save_Dir(PATH, time):
    path_to_return = os.path.join(PATH, 'save_model_{}'.format(time)) + '/'
    if not os.path.exists(path_to_return):
        os.mkdir(path_to_return)   
    return path_to_return
   
    
if __name__ == "__main__":
    if not os.path.exists('./' + PATH_LOG):
        os.mkdir(os.path.join(os.getcwd().replace('\\', '/'), PATH_LOG))
    if not os.path.exists('./' + PATH_MODEL):
        os.mkdir(os.path.join(os.getcwd().replace('\\', '/'), PATH_MODEL))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 8001))

    device = torch.device('cuda')

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_size = 21    # env.action_space.n # Steering and Throttle

    train_log = log_File_path(PATH_LOG)
    PATH_ = Model_save_Dir(PATH_MODEL, time_Feature)
    agent = DQNAgent(state_size, action_size, device)
    episodes = []

    if not agent.train:
        print("Now we load the saved model")
        agent.load_model('./save_Model/save_model_1633168898/save_model_298.h5')
    elif agent.train_from_checkpoint:
        agent.load_model(CHECK_POINT_TRAIN_PATH)
        print(f'Now we load checkpoints for continue training:  {CHECK_POINT_TRAIN_PATH.split("/")[-1]}')
    else:
        print('Thread Ready!!!')
    done = 0

    # 增加yolo目标检测算法支持
    torch.hub.set_dir('./')
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/nc1_car.pt').to(device)

    for e in range(EPISODES):      
        print("Episode: ", e)
        # Multi Thread
        if done == 2:
            print("new continued epicode!")
            done = 0
            UnityReset = 1
            episode_len = 0
        else:
            # 后期重置进入第一次recv
            print('done value:', done)
            print("new fresh episode!")
            done = 1
            s_t, v_ego_t, v_ego, v_lead, remoteHost, remotePort = Recv_data_Format(unity_Block_size, done)
            done = 0
            UnityReset = 0
            episode_len = 0

        while done == 0:
            start_time = time.time()
            if agent.t % 1000 == 0:
                rewardTot = []
            episode_len, action, time_cost, UnityReset = Send_data_Format(remoteHost, remotePort, s_t, v_ego_t, episode_len, UnityReset)
            reward, done, gap, v_ego1, v_lead, a_ego1, v_ego_1, s_t1, v_ego_t1 = Recv_data_Format(unity_Block_size, done, v_ego, v_lead, action, episode_len, s_t, v_ego_t)
            rewardTot.append(reward)
            start_count_time = int(round(time.time() * 1000))
            
            if agent.train:
                # s_t, v_ego_t, s_t1, v_ego_t1 = random_sample(s_t, v_ego_t, s_t1, v_ego_t1)
                agent.replay_memory(s_t, v_ego_t, np.argmax(linear_bin(action)), reward, s_t1, v_ego_t1, done)
                agent.train_replay()

            s_t = s_t1
            v_ego_t = v_ego_t1
            v_ego = v_ego_1
            agent.t = agent.t + 1

            print("EPISODE",  e, "TIMESTEP", agent.t, "/ ACTION", action, "/ REWARD", format(reward, '.4f'), "Avg REWARD:",
                  sum(rewardTot)/len(rewardTot), "/ EPISODE LENGTH", episode_len, "/ Q_MAX ",
                  agent.max_Q, "/ time ", time_cost, a_ego1)
            # format_str = ('EPISODE: %d TIMESTEP: %d EPISODE_LENGTH: %d ACTION: %.4f REWARD: %.4f Avg_REWARD: %.4f training_Loss: %.4f Q_MAX: %.4f gap: %.4f  v_ego: %.4f v_lead: %.4f time: %.0f a_ego: %.4f')
            # text = (format_str % (e, agent.t, episode_len, action, reward, sum(rewardTot)/len(rewardTot), agent.trainingLoss*1e3, agent.max_Q, gap, v_ego1, v_lead, time.time()-start_time, a_ego1))
            # print_out(train_log, text)
            format_str = f'EPISODE: {e} TIMESTEP: {agent.t} EPISODE_LENGTH: {episode_len} ' \
                         f'ACTION: {action:.4f} REWARD: {reward:.4f} ' \
                         f'Avg_REWARD: {sum(rewardTot) / len(rewardTot):.4f} train_loss: {agent.trainingLoss:.4f} ' \
                         f'Qmax: {agent.max_Q:.4f} gap: {gap:.4f} ' \
                         f'v_ego: {v_ego1:.4f} v_lead: {v_lead:.4f} a_ego: {a_ego1:.4f} '
            print_out(train_log, format_str)

            if done:
                agent.update_target_model()
                episodes.append(e)
                # Save model for every 2 episode
                if agent.train and (e % 2 == 0):
                    agent.save_model(os.path.join(PATH_, "save_model_{}.h5".format(e)))
                print("episode:", e, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon, " episode length:", episode_len)
                if done == 1: 
                    reset()
                    time.sleep(0.5)
            print('Data receive from unity, time:', int(round(time.time() * 1000) - start_count_time))
