from replay_buffer import replay_buffer
from net import ATNetwork, Recover_CNN, Discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import yaml
import random
import os
import setproctitle

def showmax(lt):
        index1 = 0                    
        max = 0                         
        for i in range(len(lt)):
            flag = 0                   
            for j in range(i+1,len(lt)): 
                if lt[j] == lt[i]:
                    flag += 1           
            if flag > max:              
                max = flag
                index1 = i
        return lt[index1]    

class gail(object):
    def __init__(self, env, file, fake_file, config_path='./config/config.yaml', eval=False):
        f = open(config_path)
        self.config = yaml.load(f)

        self.env = env
        self.episode = self.config['episode']
        self.capacity = self.config['capacity']
        self.gamma = self.config['gamma']
        self.lam = self.config['lam']
        self.value_learning_rate = self.config['value_learning_rate']
        self.policy_learning_rate = self.config['policy_learning_rate']
        self.discriminator_learning_rate = self.config['discriminator_learning_rate']
        self.batch_size = self.config['batch_size']
        self.policy_iter = self.config['policy_iter']
        self.disc_iter = self.config['disc_iter']
        self.value_iter = self.config['value_iter']
        self.epsilon = self.config['epsilon']
        self.entropy_weight = self.config['entropy_weight']
        self.train_iter = self.config['train_iter']
        self.clip_grad = self.config['clip_grad']
        self.file = file
        self.fake_file = fake_file
        self.action_dim = 4

        self.total_locations = self.config['total_locations']
        self.time_scale = self.config['time_scale']
        self.loc_embedding_dim = self.config['loc_embedding_dim']
        self.tim_embedding_dim = self.config['tim_embedding_dim']
        self.embedding_net = self.config['embedding_net']
        self.embedding_dim = self.loc_embedding_dim + self.tim_embedding_dim
        self.hidden_dim = self.config['hidden_dim']
        self.bidirectional = self.config['bidirectional']
        self.linear_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.device = 'cpu'
        self.data = self.config['data']
        self.starting_sample = self.config['starting_sample']
        self.starting_dist = self.config['starting_dist']

        self.act_embedding_dim = self.config['act_embedding_dim']
        self.recover_cnn_reward_ratio = self.config['recover_cnn_reward_ratio']
        self.recover_cnn_learning_rate = self.config['recover_cnn_learning_rate']

        self.model_save_interval = self.config['model_save_interval']
        self.eval = eval
        self.test_data_path = self.config['test_data_path']
        self.test_data = np.loadtxt(self.test_data_path)
        self.pre_pos_count_embedding_dim=8
        self.stay_time_embedding_dim=8

        self.policy_net = ATNetwork(
            self.total_locations,
            self.time_scale,
            self.embedding_net,
            self.loc_embedding_dim,
            self.tim_embedding_dim,
            self.pre_pos_count_embedding_dim,
            self.stay_time_embedding_dim,
            self.hidden_dim,
            self.bidirectional,
            self.data,
            self.device,
            self.starting_sample,
            self.starting_dist,
            return_prob=True
        ).to(self.device)

        self.value_net = ATNetwork(
            self.total_locations,
            self.time_scale,
            self.embedding_net,
            self.loc_embedding_dim,
            self.tim_embedding_dim,
            self.pre_pos_count_embedding_dim,
            self.stay_time_embedding_dim,
            self.hidden_dim,
            self.bidirectional,
            self.data,
            self.device,
            self.starting_sample,
            self.starting_dist,
            return_prob=False
        ).to(self.device)

        self.discriminator = Discriminator(
            self.total_locations,
            self.time_scale,
            self.loc_embedding_dim,
            self.tim_embedding_dim,
            self.pre_pos_count_embedding_dim,
            self.stay_time_embedding_dim,
            self.act_embedding_dim,
            self.hidden_dim,
        ).to(self.device)

        self.recover_cnn = Recover_CNN(
        ).to(self.device)

        self.buffer = replay_buffer(self.capacity, self.gamma, self.lam)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.000003, weight_decay=0.00001)
        self.policy_pretrain_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0003)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=0.000003, weight_decay=0.00001)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.000003, weight_decay = 0.00001)
        self.recover_cnn_optimizer = torch.optim.Adam(self.recover_cnn.parameters(), lr=0.000003, weight_decay = 0.00001)
        #loss
        self.disc_loss_func = nn.BCELoss()
        self.weight_custom_reward = None

        os.makedirs('./models/', exist_ok=True)
        os.makedirs('./eval/', exist_ok=True)

        if self.eval:
            self.policy_net.load_state_dict(torch.load('./models/policy_net.pkl'))
            self.value_net.load_state_dict(torch.load('./models/value_net.pkl'))
            self.recover_cnn.load_state_dict(torch.load('./models/recover_cnn.pkl'))
            self.discriminator.load_state_dict(torch.load('./models/discriminator.pkl'))

    def sample_real_data(self, start_index=None):
        print(self.file.shape[1])
        total_track_num = self.file.shape[0] * (self.file.shape[1] - 1)
        if start_index is None:
            sample_index = list(np.random.choice(total_track_num, self.batch_size))
            sample_index = [(x // (self.file.shape[1] - 1), x % (self.file.shape[1] - 1)) for x in sample_index]
        else:
            sample_index = list(range(start_index, start_index + self.batch_size))
            sample_index = [x % total_track_num for x in sample_index]
            sample_index = [(x // (self.file.shape[1] - 1) , x % (self.file.shape[1] - 1)) for x in list(range(start_index, start_index + self.batch_size))]
            #print(sample_index)
        time = [index[1] for index in sample_index]
        pos = [self.file[index] for index in sample_index]
        next_pos = [self.file[index[0], index[1] + 1] for index in sample_index]
        history_pos = [list(self.file[index[0], :index[1] + 1]) for index in sample_index]
        pre_pos_count = [len(list(set(hp))) for hp in history_pos]
        stay_time = []
        traj = [list(self.file[index[0]]) for index in sample_index]
        print(traj)
        for i in range(self.batch_size):
            st = 0
            for p in reversed(history_pos[i]):
                if p == next_pos[i]:
                    st += 1
                else:
                    break
            stay_time.append(st)
        home_pos = [showmax(list(self.file[index[0], 17:46])) for index in sample_index]
 
        #print(home_pos)
        action = []
        for i in range(self.batch_size):
            if next_pos[i] == pos[i] and next_pos[i] != home_pos[i]:
                action.append(0)
            elif next_pos[i] == home_pos[i]:
                action.append(2)
            elif next_pos[i] in history_pos[i] and next_pos[i] != home_pos[i]:
                action.append(3)
            else:
                action.append(1)
        return list(zip(pos, time)), action, pre_pos_count, stay_time, home_pos, traj


    def sample_fake_data(self, start_index=None):
        total_track_num = self.fake_file.shape[0] * (self.fake_file.shape[1] - 1)
        if start_index is None:
            sample_index = list(np.random.choice(total_track_num, self.batch_size))
            sample_index = [(x // (self.fake_file.shape[1] - 1), x % (self.fake_file.shape[1] - 1)) for x in sample_index]
        else:
            sample_index = list(range(start_index, start_index + self.batch_size))
            sample_index = [x % total_track_num for x in sample_index]
            sample_index = [(x // (self.fake_file.shape[1] - 1), x % (self.fake_file.shape[1] - 1)) for x in list(range(start_index, start_index + self.batch_size))]
            #print(sample_index)
        time = [index[1] for index in sample_index]
        pos = [self.fake_file[index] for index in sample_index]
        next_pos = [self.fake_file[index[0], index[1] + 1] for index in sample_index]
        history_pos = [list(self.fake_file[index[0], :index[1] + 1]) for index in sample_index]
        pre_pos_count = [len(list(set(hp))) for hp in history_pos]
        stay_time = []

        for i in range(self.batch_size):
            st = 0
            for p in reversed(history_pos[i]):
                if p == next_pos[i]:
                    st += 1
                else:
                    break
            stay_time.append(st)
        home_pos = [self.fake_file[index[0], 18] for index in sample_index]
        action = []
        for i in range(self.batch_size):
            if next_pos[i] == pos[i] and next_pos[i] != home_pos[i]:
                action.append(0)
            elif next_pos[i] == home_pos[i]:
                action.append(2)
            elif next_pos[i] in history_pos[i] and next_pos[i] != home_pos[i]:
                action.append(3)
            else:
                action.append(1)
        return list(zip(pos, time)), action, pre_pos_count, stay_time
    

    def ppo_pretrain(self, start_index):
        expert_batch = self.sample_real_data(start_index)
        expert_observations, expert_actions, expert_pre_pos_count, expert_stay_time, expert_home_pos= expert_batch[0], expert_batch[1], expert_batch[2], expert_batch[3], expert_batch[4]
        expert_observations = np.vstack(expert_observations)
        expert_observations = torch.LongTensor(expert_observations).to(self.device)
        expert_actions_index = torch.LongTensor(expert_actions).unsqueeze(1).to(self.device)

        expert_stay_time = torch.LongTensor(expert_stay_time).unsqueeze(1).to(self.device)
        expert_pre_pos_count = torch.LongTensor(expert_pre_pos_count).unsqueeze(1).to(self.device)
        expert_home_pos = torch.LongTensor(expert_home_pos).unsqueeze(1).to(self.device)        
        expert_trajs = torch.cat([expert_observations, expert_pre_pos_count, expert_stay_time], 1)

        probs = self.policy_net.forward(expert_trajs[:, 0], expert_trajs[:, 1], expert_trajs[:, 2], expert_trajs[:, 3]).log()
        #print(probs)

        loss = F.nll_loss(probs, expert_actions_index.squeeze())
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_pretrain_optimizer.step()


    def ppo_train(self):
        pos, times, history_pos, home_point, pre_pos_count, stay_time, actions, returns, advantages, traj = self.buffer.sample(self.batch_size)
        pos = torch.LongTensor(pos).to(self.device)
        times = torch.LongTensor(times).to(self.device)
        history_pos = torch.LongTensor(history_pos).to(self.device)
        home_point = torch.LongTensor(home_point).to(self.device)
        stay_time = torch.LongTensor(stay_time).to(self.device)
        pre_pos_count = torch.LongTensor(pre_pos_count).to(self.device)  
        traj = torch.LongTensor(traj).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        advantages = (advantages - advantages.mean()) / advantages.std()
        advantages = advantages.detach()
        returns = torch.FloatTensor(returns).to(self.device).unsqueeze(1).detach()

        for _ in range(self.value_iter):
            values = self.value_net.forward(pos, times, pre_pos_count, stay_time, home_point)
            value_loss = (returns - values).pow(2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        actions_d = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        old_probs = self.policy_net.forward(pos, times, pre_pos_count, stay_time, home_point)

        old_probs = old_probs.gather(1, actions_d)
        dist = torch.distributions.Categorical(old_probs)
        entropy = dist.entropy().unsqueeze(1)
        for _ in range(self.policy_iter):
            probs = self.policy_net.forward(pos, times, pre_pos_count, stay_time, home_point)
            probs = probs.gather(1, actions_d)
            ratio = probs / old_probs.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * advantages
            policy_loss = - torch.min(surr1, surr2) - self.entropy_weight * entropy
            policy_loss = policy_loss.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_grad)
            self.policy_optimizer.step()


    def discriminator_pretrain(self):
        expert_batch = self.sample_real_data()
        expert_observations, expert_actions, expert_pre_pos_count, expert_stay_time = expert_batch[0], expert_batch[1], expert_batch[2], expert_batch[3]
        expert_observations = np.vstack(expert_observations)
        expert_observations = torch.LongTensor(expert_observations).to(self.device)
        expert_actions_index = torch.LongTensor(expert_actions).unsqueeze(1).to(self.device)
        
        expert_stay_time = torch.LongTensor(expert_stay_time).unsqueeze(1).to(self.device)
        expert_pre_pos_count = torch.LongTensor(expert_pre_pos_count).unsqueeze(1).to(self.device)
        
        expert_trajs = torch.cat([expert_observations, expert_actions_index, expert_pre_pos_count, expert_stay_time], 1)
        expert_labels = torch.FloatTensor(self.batch_size, 1).fill_(0.0).to(self.device)

        fake_batch = self.sample_fake_data()
        fake_observations, fake_actions, fake_pre_pos_count, fake_stay_time = fake_batch[0], fake_batch[1], fake_batch[2], fake_batch[3]
        fake_observations = np.vstack(fake_observations)
        fake_observations = torch.LongTensor(fake_observations).to(self.device)
        fake_actions_index = torch.LongTensor(fake_actions).unsqueeze(1).to(self.device)
        
        fake_stay_time = torch.LongTensor(fake_stay_time).unsqueeze(1).to(self.device)
        fake_pre_pos_count = torch.LongTensor(fake_pre_pos_count).unsqueeze(1).to(self.device)
        
        fake_trajs = torch.cat([fake_observations, fake_actions_index, fake_pre_pos_count, fake_stay_time], 1)

        labels = torch.FloatTensor(self.batch_size, 1).fill_(1.0).to(self.device)

        # * optimize discriminator
        expert_reward = self.discriminator.forward(expert_trajs[:, 0], expert_trajs[:, 1], expert_trajs[:, 2], expert_trajs[:, 3], expert_trajs[:, 4])
        current_reward = self.discriminator.forward(fake_trajs[:, 0], fake_trajs[:, 1], fake_trajs[:, 2], fake_trajs[:, 3], fake_trajs[:, 4])
        expert_loss = self.disc_loss_func(expert_reward, expert_labels)
        current_loss = self.disc_loss_func(current_reward, labels)
        
        loss = (expert_loss + current_loss) / 2
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()


    def discriminator_train(self):
        expert_batch = self.sample_real_data()
        expert_observations, expert_actions, expert_pre_pos_count, expert_stay_time, expert_home_pos, expert_traj = expert_batch[0], expert_batch[1], expert_batch[2], expert_batch[3], expert_batch[4], expert_batch[5]
        expert_observations = np.vstack(expert_observations)
        expert_observations = torch.LongTensor(expert_observations).to(self.device)
        expert_actions_index = torch.LongTensor(expert_actions).unsqueeze(1).to(self.device)
        
        expert_stay_time = torch.LongTensor(expert_stay_time).unsqueeze(1).to(self.device)
        expert_pre_pos_count = torch.LongTensor(expert_pre_pos_count).unsqueeze(1).to(self.device)
        expert_home_pos = torch.LongTensor(expert_home_pos).unsqueeze(1).to(self.device)
        expert_traj = torch.LongTensor(expert_traj).to(self.device)
        expert_trajs = torch.cat([expert_observations, expert_actions_index, expert_pre_pos_count, expert_stay_time, expert_traj], 1)
        expert_labels = torch.FloatTensor(self.batch_size, 1).fill_(0.0).to(self.device)

        pos, times, history_pos, home_point, pre_pos_count, stay_time, actions, _, _,traj = self.buffer.sample(self.batch_size)
        pos = torch.LongTensor(pos).view(-1, 1).to(self.device)
        times = torch.LongTensor(times).view(-1, 1).to(self.device)
        observations = torch.cat([pos, times], dim=-1).to(self.device)

        actions_index = torch.LongTensor(actions).unsqueeze(1).to(self.device)

        stay_time = torch.LongTensor(stay_time).unsqueeze(1).to(self.device)
        pre_pos_count = torch.LongTensor(pre_pos_count).unsqueeze(1).to(self.device)
        history_pos = torch.LongTensor(history_pos).to(self.device)
        traj = torch.LongTensor(traj).to(self.device)
        trajs = torch.cat([observations, actions_index, pre_pos_count, stay_time,  traj], 1)

        labels = torch.FloatTensor(self.batch_size, 1).fill_(1.0).to(self.device)

        for _ in range(self.disc_iter):
            # * optimize recover
            '''
            expert_reward = self.recover_cnn.forward(expert_traj[:, 5])
            current_reward = self.recover_cnn.forward(traj[:, 5])
            expert_loss = self.disc_loss_func(expert_reward, expert_labels)
            current_loss = self.disc_loss_func(current_reward, labels)


            loss = (expert_loss + current_loss) / 2

            self.recover_cnn_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.recover_cnn.parameters(), self.clip_grad)
            self.recover_cnn_optimizer.step()
            '''

            # * optimize discriminator
            expert_reward = self.discriminator.forward(expert_trajs[:, 0], expert_trajs[:, 1], expert_trajs[:, 2], expert_trajs[:, 3], expert_trajs[:, 4])
            current_reward = self.discriminator.forward(trajs[:, 0], trajs[:, 1], trajs[:, 2], trajs[:, 3], trajs[:, 4])
            expert_loss = self.disc_loss_func(expert_reward, expert_labels)
            current_loss = self.disc_loss_func(current_reward, labels)

            loss = (expert_loss + current_loss) / 2
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad)
            self.discriminator_optimizer.step()


    def get_reward(self, pos, time, action, pre_pos_count, stay_time):
        pos = torch.LongTensor([pos]).to(self.device)
        time = torch.LongTensor([time]).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        pre_pos_count = torch.LongTensor([pre_pos_count]).to(self.device)
        stay_time = torch.LongTensor([stay_time]).to(self.device)

        d_reward = self.discriminator.forward(pos, time, action, pre_pos_count, stay_time)
        reward = d_reward
        log_reward = - reward.log()
        return log_reward.detach().item()


    def get_r_reward(self, traj):
        traj = torch.LongTensor([traj]).to(self.device)
        r_reward = self.recover_cnn.forward(traj)
        reward =  r_reward 
        log_reward = - reward.log()
        return log_reward


    def eval_test(self, index):
        result = np.zeros_like(self.test_data)
        for i in range(len(self.test_data)):
            t = 0
            pos = self.test_data[i][t] - 1
            home_point = showmax(list(self.test_data[i][17:46])) 
            #print(home_point)
            history_pos = []
            stay_time = 0
            pre_pos_count = 1
            self.env.set_state(pos=int(pos), t=int(t))
            result[i][t] = pos
            while True:
                action = self.policy_net.act(torch.LongTensor(np.expand_dims(pos, 0)).to(self.device), torch.LongTensor(np.expand_dims(t, 0)).to(self.device), torch.LongTensor(np.expand_dims(pre_pos_count, 0)).to(self.device), torch.LongTensor(np.expand_dims(stay_time, 0)).to(self.device))
                next_pos, next_t, done, history_pos, home_point, pre_pos_count, stay_time = self.env.step(action, history_pos, home_point, True)
                result[i][next_t] = next_pos
                pos = next_pos
                t = next_t
                if done:
                    break
        np.save('./eval/eval_{}.npy'.format(index), result.astype(np.int))

    def generator_pretrain_run(self, max_epoch):
        epoch = 0
        start_index = 0
        while epoch <= max_epoch:
            self.ppo_pretrain(start_index)
            start_index += 1
            epoch += 1
            start_index = start_index % (self.file.shape[0] * (self.file.shape[1] - 1))
            if start_index > (self.file.shape[0] * (self.file.shape[1] - 1) - self.batch_size):
                start_index = 0


    def discriminator_pretrain_run(self, times):
        for t in range(times):
            self.discriminator_pretrain()
            
    def run(self):
        setproctitle.setproctitle('gail')
        print(self)
        self.generator_pretrain_run(10000)#generator pretrain setting
        print('generator')
        self.discriminator_pretrain_run(10000)#discriminator pretrain setting
        print('pretrain end')

        for i in range( self.episode):
            pos, time, history_pos, home_point, pre_pos_count, stay_time = self.env.reset()
            total_custom_reward =0
            while True:
                action = self.policy_net.act(torch.LongTensor(np.expand_dims(pos, 0)).to(self.device), torch.LongTensor(np.expand_dims(time, 0)).to(self.device), torch.LongTensor(np.expand_dims(pre_pos_count, 0)).to(self.device), torch.LongTensor(np.expand_dims(stay_time, 0)).to(self.device))
                next_pos, next_time, done, next_history_pos, next_home_point, next_pre_pos_count, next_stay_time= self.env.step(action, history_pos, home_point)
                custom_reward = self.get_reward(pos, time, action, next_pre_pos_count, next_stay_time)

                value = self.value_net.forward(torch.LongTensor(np.expand_dims(pos, 0)).to(self.device), torch.LongTensor(np.expand_dims(time, 0)).to(self.device), torch.LongTensor(np.expand_dims(pre_pos_count, 0)).to(self.device), torch.LongTensor(np.expand_dims(stay_time, 0)).to(self.device)).detach().item()
                #print(value)
                self.buffer.store(pos, time, history_pos, home_point, pre_pos_count, stay_time, action, custom_reward, done, value, traj)
                total_custom_reward += custom_reward
                pos = next_pos
                time = next_time
                history_pos = next_history_pos
                home_point = next_home_point
                pre_pos_count = next_pre_pos_count
                stay_time = next_stay_time

                if done:
                    r_reward = self.get_r_reward(history_pos)
                    total_custom_reward + 0.0001*r_reward
                    if len(history_pos)==1:
                        break
                    if not self.weight_custom_reward:
                        self.weight_custom_reward = total_custom_reward
                    else:
                        self.weight_custom_reward = 0.99 * self.weight_custom_reward + 0.01 * total_custom_reward
                    if len(self.buffer) >= self.train_iter:
                        self.buffer.process()
                        self.discriminator_train()
                        self.ppo_train()
                        self.buffer.clear()

                    print('episode: {}  custom_reward: {:.3f}  weight_custom_reward: {:.4f}'.format(i + 1, total_custom_reward, self.weight_custom_reward))
                    break
            if (i + 1) % self.model_save_interval == 0:
                torch.save(self.policy_net.state_dict(), './models/policy_net.pkl')
                torch.save(self.value_net.state_dict(), './models/value_net.pkl')
                torch.save(self.recover_cnn.state_dict(), './models/recover_cnn.pkl')
                torch.save(self.discriminator.state_dict(), './models/discriminator.pkl')
                self.eval_test((i + 1) // self.model_save_interval)
