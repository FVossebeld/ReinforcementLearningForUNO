import sys
import threading
import numpy as np
import keras
from agent import UnoAgent
from environment import UnoEnvironment

PLAYER_COUNT = 4
COLLECTOR_THREADS = 1
INITIAL_EPSILON = 1
EPSILON_DECAY = 0.999975
MIN_EPSILON = 0.01

def run(agent):
    # initialize environment
    epsilon = INITIAL_EPSILON
    env = UnoEnvironment(PLAYER_COUNT)

    model1 = keras.models.load_model('Agents/Agent_B.h5')
    model2 = keras.models.load_model('Agents/Agent_C.h5')
    model3 = keras.models.load_model('Agents/Agent_D.h5')

    # # Training Scenario 2:
    # models = [model1, model2, model3]

    counter = 0
    while True:
        done = False
        state = None

        # # Training Scenario 3: Randomly choose a model for the opponents each game
        # opponent_model = np.random.choice([model1, model2, model3])
        # models = [opponent_model] * 3  # Use the chosen model for all three opponents

        # # Training Scenario 4: Randomly choose a model for each opponent
        # models = [np.random.choice(all_models) for _ in range(3)]

        rewards = []
        # run one episode
        while not done:

            active_player = env.current_player()
            if state is None or np.random.sample() < epsilon or not agent.initialized:
                # choose a random action
                # Get all possible actions
                all_actions = range(env.action_count())
                # Filter to get only legal actions
                legal_actions = [action for action in all_actions if env.legal_move(action)]
                # Randomly choose from the legal actions
                action = np.random.choice(legal_actions)
            else:
                # choose an action from the policy.
                # Currently, hardcoded for four players. Would be nice to make it more dynamic.
                if active_player == 0:
                    predicted_Q = agent.predict(state) * env.get_legal_cards()
                else:
                    predicted_Q = agent.predict_special(models[active_player - 1], state) * env.get_legal_cards()

                if np.sum(predicted_Q) == 0: # When all legal moves have a Q value of 0
                    print("All legal moves have Q-values of 0. Choosing a random action.")
                    # Get all possible actions
                    all_actions = range(env.action_count())
                    # Filter to get only legal actions
                    legal_actions = [action for action in all_actions if env.legal_move(action)]
                    # Randomly choose from the legal actions
                    action = np.random.choice(legal_actions)
                else:
                    action = np.argmax(predicted_Q)
                if action == 0 and not env.legal_move(action): # When an illegal action of zero is chosen
                    # choose a random action
                    # Get all possible actions
                    all_actions = range(env.action_count())
                    # Filter to get only legal actions
                    legal_actions = [action for action in all_actions if env.legal_move(action)]
                    # Randomly choose from the legal actions
                    action = np.random.choice(legal_actions)

                        
            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            if state is not None and active_player==0:
                # include the current transition in the replay memory
                agent.update_replay_memory((state, action, reward, new_state, done))
            state = new_state

            if agent.initialized:
                # decay epsilon
                epsilon *= EPSILON_DECAY
                epsilon = max(epsilon, MIN_EPSILON)

        # log metrics
        agent.logger.scalar('cumulative_reward', np.sum(rewards))
        agent.logger.scalar('mean_reward', np.mean(rewards))
        agent.logger.scalar('game_length', len(rewards))
        agent.logger.scalar('epsilon', epsilon)

        # reset the environment for the next episode
        env.reset()

        
if __name__ == '__main__':
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # initialize the training agent
    dummy_env = UnoEnvironment(1)
    agent = UnoAgent(dummy_env.state_size(), dummy_env.action_count(), model_path)
    del dummy_env

    # start up threads for experience collection
    for _ in range(COLLECTOR_THREADS):
        threading.Thread(target=run, args=(agent,), daemon=True).start()

    # blocking call to agent, invoking an endless training loop
    agent.train()
