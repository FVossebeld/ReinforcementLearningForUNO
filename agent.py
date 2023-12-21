import os
import random
import collections
import numpy as np
from my_tensorboard import CometLogger  # Modify this as per your Tensorboard logger for PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from comet_ml import Experiment

class UnoAgent:
    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 512
    DISCOUNT_FACTOR = 0.7
    MODEL_UPDATE_FREQUENCY = 20
    MODEL_SAVE_FREQUENCY = 1000

    def __init__(self, state_size, action_count, model_path=None):
        print('Initializing agent...')
        self.initialized = False
        self.logger = CometLogger(api_key="HRHycl4Fy5Dt581uHTYKXEBwU", project_name="uno-bot",)
        print("Num GPUs Available: ", torch.cuda.device_count())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            print('Creating model...')
            self.model = self.create_model(state_size, action_count).to(self.device)
            self.target_model = self.create_model(state_size, action_count).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            print('Loading model to continue the training process...')
            self.model = torch.load(model_path).to(self.device)
            self.target_model = torch.load(model_path).to(self.device)

        self.replay_memory = collections.deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.optimizer = optim.Adam(self.model.parameters())

    def create_model(self, input_size, output_size):
        model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array(state)).float().to(self.device)
            return torch.argmax(self.model(state_tensor)).item()

    def train(self):
        counter = 0
        while True:
            if len(self.replay_memory) < self.BATCH_SIZE:
                continue

            minibatch = random.sample(self.replay_memory, self.BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            next_q_values = self.model(next_states).max(1)[0]
            expected_q_values = rewards + self.DISCOUNT_FACTOR * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, expected_q_values.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.scalar('loss', loss.item())

            counter += 1
            if counter % self.MODEL_UPDATE_FREQUENCY == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                if not self.initialized:
                    print('Agent initialized')
                    self.initialized = True

            if counter % self.MODEL_SAVE_FREQUENCY == 0:
                folder = f'models/{self.logger.timestamp}'
                os.makedirs(folder, exist_ok=True)
                torch.save(self.model.state_dict(), f'{folder}/model-{counter}.pt')
        self.logger.flush()
        
