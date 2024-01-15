from my_tensorboard import CometLogger
import os
import random
import collections
import numpy as np
from keras import models, layers, optimizers

class UnoAgent:

    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 512
    DISCOUNT_FACTOR = 0.95
    MODEL_UPDATE_FREQUENCY = 20
    MODEL_SAVE_FREQUENCY = 500

    def __init__(self, state_size, action_count, model_path=None):
        print('Initializing agent...')
        self.initialized = False
        self.logger = CometLogger(api_key="HRHycl4Fy5Dt581uHTYKXEBwU", project_name="uno-bot",)

        if model_path is None:
            print('Creating model...')
            # initialize the prediction model and a clone of it, the target model
            self.model = self.create_model(state_size, action_count)
            self.target_model = self.create_model(state_size, action_count)
            self.target_model.set_weights(self.model.get_weights())
        else:
            print('Loading model to continue the training process...')
            # load existing model to continue training
            self.model = models.load_model(model_path)
            self.target_model = models.load_model(model_path)

        # initialize the replay memory
        self.replay_memory = collections.deque(maxlen=self.REPLAY_MEMORY_SIZE)

    def create_model(self, input_size, output_size):
        # define the model architecture
        model = models.Sequential()
        model.add(layers.Dense(units=64, activation='relu', input_shape=(input_size,)))
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=output_size, activation='linear'))

        # compile the model
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def predict_special(self, temp_model, state):
        print("special predicting")
        return temp_model.predict(np.array(state).reshape(-1, *state.shape), verbose=None)[0]

    def update_replay_memory(self, transition):
        # add a state transition to the replay memory
        self.replay_memory.append(transition)

    def predict(self, state):
        print("predicting")
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=None)[0]


    def train(self):
        counter = 0
        while True:
            # Ensure there is enough data in replay memory
            if len(self.replay_memory) < self.BATCH_SIZE:
                continue

            # Sample a minibatch from the replay memory
            minibatch = random.sample(self.replay_memory, self.BATCH_SIZE)

            # Separate states, actions, rewards, next_states, and dones
            states = np.array([transition[0] for transition in minibatch])
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = np.array([transition[3] for transition in minibatch])
            dones = np.array([transition[4] for transition in minibatch])

            # Predict Q-values for starting states
            q_values = self.model.predict(states)
            # Predict future Q-values for next states
            future_q_values = self.model.predict(next_states)
            # Max future Q value for each next state
            max_future_q = np.max(future_q_values, axis=1)

            # Update Q-values for each action taken
            for i in range(self.BATCH_SIZE):
                if dones[i]:
                    q_values[i, actions[i]] = rewards[i]
                else:
                    q_values[i, actions[i]] +=  self.DISCOUNT_FACTOR * max_future_q[i]

            # Train the model on the minibatch
            hist = self.target_model.fit(x=states, y=q_values, batch_size=self.BATCH_SIZE, verbose=0)
            self.logger.scalar('loss', hist.history['loss'][0])
            self.logger.scalar('accuracy',  hist.history['accuracy'][0])  # Make sure this key matches your model's compile metrics

            counter += 1
            # Update target model
            if counter % self.MODEL_UPDATE_FREQUENCY == 0:
                self.model.set_weights(self.target_model.get_weights())
                if not self.initialized:
                    print('Agent initialized')
                    self.initialized = True

            # Save model periodically
            if counter % self.MODEL_SAVE_FREQUENCY == 0:
                folder = f'models/{self.logger.timestamp}'
                os.makedirs(folder, exist_ok=True)
                self.model.save(f'{folder}/model-{counter}.h5')