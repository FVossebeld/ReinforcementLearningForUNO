import sys
import time
import keras
import random
import numpy as np
import pandas as pd
from environment import UnoEnvironment
from renderer import *
from keras.models import load_model


def game(models, model_names):
    MOVE_TIME = 0


    # extract player types (0 for AI, 1 for AI)
    player_types = ['AI-1', 'AI-2', 'AI-3', 'AI-4']
    player_names = model_names
    player_moves = [0, 0, 0, 0]


    # check if AI player is present
    if 0 in player_types:
        if models is not None:
            print('Loading model...')
        else:
            print('Please specify a model path.')
            exit()

    # initialize game variables
    game_messages = []
    game_winners = []
    game_moves = []
    last_move = time.time()

    print('Initializing game environment...')
    env = UnoEnvironment(len(player_types))

    done = False
    game_finished = False

    print('Done! Running game loop...')
    while not done:
        if not game_finished:
            # game logic
            if time.time() - last_move < MOVE_TIME:
                # wait until move delay is reached
                action = None
            else:
                # AI player
                state = env.get_state()
                active_player = env.current_player()
                model = models[active_player - 1]
                predicted_Q = model.predict(state.reshape((1, -1)))[0] * env.get_legal_cards()

                if np.sum(predicted_Q) == 0:  # When all legal moves have a Q value of 0
                    # Get all possible actions
                    all_actions = range(env.action_count())
                    # Filter to get only legal actions
                    legal_actions = [action for action in all_actions if env.legal_move(action)]
                    # Randomly choose from the legal actions
                    action = np.random.choice(legal_actions)
                else:
                    action = np.argmax(predicted_Q)

                if action == 0 and not env.legal_move(action):  # When an illegal action of zero is chosen
                    # choose a random action
                    # Get all possible actions
                    all_actions = range(env.action_count())
                    # Filter to get only legal actions
                    legal_actions = [action for action in all_actions if env.legal_move(action)]
                    # Randomly choose from the legal actions
                    action = np.random.choice(legal_actions)

                # make random move if the AI selected an illegal move
                if not env.legal_move(action):
                    game_messages.append((time.time(), f'{player_names[env.turn]} selected an illegal action, play random card.'))
                    while not env.legal_move(action):
                        action = np.random.randint(env.action_count())
            

            if action is not None:
                # play the selected action
                _, _, game_finished, step_info = env.step(action)
                last_move = time.time()

                turn = step_info['turn']
                player_moves[turn] += 1
                player_status = step_info['player']

            # check if the current player is out of the game
                if player_status == -1 or player_status == 2:
                    if player_status == -1:
                        game_messages.append((time.time(), f'{player_names[turn]} eliminated due to illegal move.'))
                    elif player_status == 2:
                        print(f'{player_names[turn]} has finished!')
                        game_winners.append(player_names[turn])
                        game_moves.append(player_moves[turn])
                    del player_types[turn]
                    del player_names[turn]
                    del player_moves[turn]

                # update game screen once after game has finished
                if game_finished:
                    game_winners.append(player_names[turn-1])
                    game_moves.append(player_moves[turn-1]+1)
                    return(game_winners, game_moves, time.time())

    done = True

    return(game_winners, game_moves, time.time())


def init_models(argv):
    # Initialize models
    model_B = keras.models.load_model('Agents/Agent_b.h5')
    model_S1 = keras.models.load_model('Agents/model-S1-8000.h5')
    model_S2 = keras.models.load_model('Agents/model-S2-8000.h5')
    model_S3 = keras.models.load_model('Agents/model-S3-8000.h5')
    model_S4 = keras.models.load_model('Agents/model-S4-8000.h5')

    all_model_names = ["B", "S1", "S2", "S3", "S4"]
    all_models = [model_B, model_S1, model_S2, model_S3, model_S4]
    
    # Initialize arrays with models that will be played with
    models = []
    model_names = []

    # Check wich models are given as user input and make sure they are used for the game
    if len(argv) != 5:
        print("no propper number of commands (of 4) are given: <model1> <model2> <model3> <model4>")
        quit
    else:
        for i in range (1, 5, 1):
            for j in range(5):
                if argv [i] == all_model_names[j]:
                    print(all_model_names[j])
                    models.append(all_models[j])
                    model_names.append(all_model_names[j])
        if len(models) != 4:
            print("no propper model names are given as input: B, S1, S2, S3, S4")
            quit

    return (models, model_names)


def shuffle_models (models, model_names):
    # Randomly arange the order of the players (for fairer games)
    indices = [0, 1, 2, 3]
    random.shuffle(indices)

    models_input = [models[i] for i in indices]
    models_names_input = [model_names[i] for i in indices]
    return (models_input, models_names_input)
    

def main():
    # Define the number of mathes being played
    matches = 2

    models, model_names = init_models(sys.argv)

    # Initialize dataframe for storage of results
    results_df = pd.DataFrame({"1st":[], "2nd":[], "3rd":[], "4th":[], 
                               "1st_moves":[], "2nd_moves":[], "3rd_moves":[], "4th_moves":[],
                               "time":[]
                               })
    
    # Itterate over all matches
    for _ in range(matches):
        
        # Shuffle the order of the models
        models_input, model_names_input = shuffle_models(models, model_names)
        
        # Play the game
        rankings, moves, time = game(models_input, model_names_input)
        
        # Append the results of the game
        new_row = {"1st":rankings[0], "2nd":rankings[1], "3rd":rankings[2], "4th":rankings[3], 
                    "1st_moves":int(moves[0]), "2nd_moves":int(moves[1]), "3rd_moves":int(moves[2]), "4th_moves":int(moves[3]),
                    "time":time
                    }
        results_df = results_df.append(new_row, ignore_index=True)

    # Write the results of the tournament to a CSV file
    results_df.to_csv('tournament_results.csv', index=True)
    print('Tournament completed. Results saved to tournament_results.csv')

    
if __name__ == '__main__':
    main()



