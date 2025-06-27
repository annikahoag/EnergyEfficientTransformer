import cma
import random
import math
import os
import csv
import numpy as np
import tqdm
import torch.nn as nn
import sqlite3
import json

from time import time
from pprint import pprint

from src.models import *
from src.training_helpers import *
from src.pot_evaluation import *
from torch.utils.data import Dataset, DataLoader, TensorDataset

os.environ['DGLBACKEND'] = 'pytorch'
import dgl


# Constants
DATA_PATH = './'  # Replace with your actual path
OUTPUT_FOLDER = os.path.join(DATA_PATH, 'Pre_processed_data')
FILE_NAME = 'power_output.csv'
DB_NAME = 'results.db'


class Config:
    def __init__(self):
        self.dataset = 'MBA'  #         # 'dataset': Specifies the dataset for model optimization and evaluation.  Change according to the dataset you're working with (e.g., SMAP, SWaT).
        self.model = 'TransNAS_TSAD'  #  'model': Designates the model to be optimized. Here, 'TransNAS_TSAD'  refers to our dynamic transformer model tailored for time-series anomaly detection.
        self.retrain = True           # 'retrain': Indicates whether the model should undergo retraining.   # Setting this to True forces the model to train from scratch, ignoring any pre-trained weights.
        self.test = False             # 'test': Controls whether the script is in testing mode. When set to False,  # the script focuses on training the model. Set to True for evaluating the model's performance.
config = Config()



# Creates SQLite database and table
def create_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS study_results (
            id INTEGER PRIMARY KEY,
            F1_value REAL, 
            params REAL,
            GPU_usage REAL,
            GPU_avg REAL,
            fitness REAL
        )
    """)
    conn.commit()
    conn.close()



# Used to scale the suggested values from CMA-ES to make them usable in our model
def scale_params(val, high, low, decimals, step):
    width = high - low # in case the starting value isn't 0
    if step != 1:
        val = val + (step-val) % step

    if decimals==0:
        return int(round(low + ((val - low) % width), decimals))
    else:
        return round(low + ((val - low) % width), decimals)



# Average the power readings we have so far
def get_power(filename):

    with open(filename, 'r') as f:
        reader = list(csv.reader(f, delimiter=','))
        num_iters = len(reader)

    total=0
    num_skips=0

    for i in range(num_iters):
       
        if reader[i][0] != '#gpu' and reader[i][0] != '#Idx':
            total = total + float(reader[i][1])
        else:
            num_skips+=1

    avg_power = total / (num_iters-num_skips)

    return total, avg_power



# Saving trial values to SQLite database
def save_trial_to_db(f1, param_count, wattage, wattage_avg, fitness):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO study_results(F1_value, params, GPU_usage, GPU_avg, fitness)
        VALUES (?, ?, ?, ?, ?)
    ''', (f1, param_count, wattage, wattage_avg, fitness))
    conn.commit()
    conn.close()



 # Objective function --> takes in a vector of parameter values and trains a model with those parameters
 # Returns fitness function as the ratio of F1 to GPU wattage, which we try to increase
def fitness_function(params):

    # Load dataset
    print("output folder: ", OUTPUT_FOLDER)
    train_loader, test_loader, labels = load_dataset(OUTPUT_FOLDER,config.dataset,config)
    print(labels.shape[1])  # Print label shape for debugging
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD


    # Set up the given parameters so that the model can be trained with them
    learning_rate = scale_params(params[0], 0.0001, 0.009, 4, 1) 
    dropout_rate = scale_params(params[1], 0.1, 0.5, 1, 1)
    dim_feedforward = scale_params(params[2], 8, 128, 0, 1)
    batch = scale_params(params[3], 16, 128, 0, 16)
    encoder_layers = scale_params(params[4], 1, 3, 0, 1)
    decoder_layers = scale_params(params[5], 1, 3, 0, 1)
    activation_function = scale_params(params[6], 1, 4, 0, 1)
    time_warping = scale_params(params[7], 0, 1, 0, 1)
    time_masking = scale_params(params[8], 0, 1, 0, 1)
    gaussian_noise = scale_params(params[9], math.log10(1e-4), math.log10(1e-1), 0, 1)
    use_linear_embedding = scale_params(params[10], 0, 1, 0, 1)
    
    if config.dataset in ['MSL', 'SMAP', 'SWaT', 'WADI', 'SMD', 'NAB', 'MBA', 'UCR']:
        phase_type = scale_params(params[11], 1, 2, 0, 1)
    else:
        phase_type = scale_params(params[11], 1, 3, 0, 1)
    
    self_conditioning = scale_params(params[12], 0, 1, 0, 1) 
    layer_norm = scale_params(params[13], 0, 1, 0, 1)
    positional_encoding_type = scale_params(params[14], 1, 2, 0, 1)
    num_ffn_layers = scale_params(params[15], 1, 3, 0, 1)
    nhead = labels.shape[1] # TODO: I'm not sure if this is right but it seemed to be less evolved and more based on dataset

    # Putting some params into the list as strings or Booleans
    if activation_function==1:
        activation_function = 'relu'
    elif activation_function==2:
        activation_function = 'leaky_relu'
    elif activation_function==3:
        activation_function = 'tanh'
    elif activation_function==4:
        activation_function = 'sigmoid'

    if time_warping==0:
        time_warping=False
    else:
        time_warping=True
    
    if time_masking==0:
        time_masking=False
    else:
        time_masking=True

    if use_linear_embedding==0:
        use_linear_embedding = False
    else:
        use_linear_embedding=True
    
    if config.dataset in ['MSL', 'SMAP', 'SWaT', 'WADI', 'SMD', 'NAB', 'MBA', 'UCR']:
        if phase_type == 1:
            phase_type = '2phase'
        elif phase_type == 2:
            phase_type = 'iterative'
    else:
        if phase_type == 1:
            phase_type = '1phase'
        elif phase_type == 2:
            phase_type = '2phase'
        elif phase_type == 3:
            phase_type = 'iterative'

    if self_conditioning==0:
        self_conditioning = False
    else:
        self_conditioning = True 
    
    if layer_norm==0:
        layer_norm = False
    else:
        layer_norm = True 
    
    if positional_encoding_type==1:
        positional_encoding_type = 'sinusoidal'
    else:
        positional_encoding_type = 'fourier'


    # The model
    final_params = {
        'lr': learning_rate,
        'dropout_rate': dropout_rate,
        'dim_feedforward': dim_feedforward,
        'batch': batch,
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers,
        'attention_type': "scaled_dot_product",
        'positional_encoding_type': positional_encoding_type,
        'phase_type': phase_type,
        'gaussian_noise_std': gaussian_noise,
        'time_warping': time_warping,
        'time_masking': time_masking,
        'self_conditioning': self_conditioning,
        'layer_norm': layer_norm,
        'activation_function': activation_function,
        'use_linear_embedding': use_linear_embedding,
        'nhead': nhead,  # Dynamic based on labels
        'num_ffn_layers': num_ffn_layers
    }


    # Load model with suggested parameters
    print("\n LOADING MODEL \n")
    model, optimizer, scheduler, epoch, accuracy_list = load_model(config.model, labels.shape[1],config, **final_params)
    model.double()
    trainD, testD = convert_to_windows(trainD, model,config), convert_to_windows(testD, model,config)

  
    trial_timeout=40;     #Pass it to pot_eval () Set it according to the size of dataset (some trials get stuck for infinite time due to irregular parameters combination)


    # Training phase
    if not config.test:
        print(f'Training {config.model} on {config.dataset}')
        num_epochs = 5
        start = time.time()

        for e in tqdm(range(epoch+1, epoch+num_epochs+1)):
            lossT, lr = optimize_model(e, model, trainD, trainO, optimizer, scheduler, config)
            accuracy_list.append((lossT, lr))
            print(f"Epoch {e}, Loss: {lossT}, Learning Rate: {lr}")
            
        print('Training time: ' + "{:10.4f}".format(time.time() - start) + ' s')


    # Testing phase
    model.eval()  # Set model to evaluation mode
    print(f'Testing {config.model} on {config.dataset}')
    loss, y_pred = optimize_model(0, model, testD, testO, optimizer, scheduler,config, training=False)


    # Initialize an empty list to collect result DataFrames
    result_list = []


    try:
        lossT, _ = optimize_model(0, model, trainD, trainO, optimizer, scheduler, config, training=False)
        for i in range(loss.shape[1]):
            lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
            result, pred = pot_eval(config,trial_timeout,lt, l, ls)
            # Add the result DataFrame to the list
            if isinstance(result, dict):
                result = pd.DataFrame([result])  # Convert dict to DataFrame
            result_list.append(result)
        df = pd.concat(result_list, ignore_index=True)
    except TimeoutError:
        print(f"Trial timed out during score calculation.")
        return {'f1': 0.0, 'num_params': float('inf')}
    except Exception as e:
        print(f"An exception occurred: {e}")
        return {'f1': 0.0, 'num_params': float('inf')}
    

    
    #------------------------Objective #1 --> Accuracy------------------------
    # Finalize results
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1).astype(int)
    num_params = sum(p.numel() for p in model.parameters())
    result, _ = pot_eval(config,trial_timeout,lossTfinal, lossFinal, labelsFinal)
    Result2 = result
    pprint(Result2)
    f1_score = result['f1']


    #----------------------Objective #3 --> GPU Usage------------------------
    total_power, avg_power = get_power(FILE_NAME)

    fitness = f1_score / avg_power
    
    # Save trial to database
    save_trial_to_db(f1_score, num_params, total_power, avg_power, fitness)

    
    # TODO: figure out exactly what the fitness function is, this is just a placeholder because it makes sense but idk if it's too complictaed/convoluted
    return -fitness



def main(): 
    
    # Create database if it doesn't already exist 
    create_db()
    print("\nSucessfully created DB\n")

    # Load dataset
    train_loader, test_loader, labels = load_dataset(OUTPUT_FOLDER,config.dataset,config)
    print(labels.shape[1])  # Print label shape for debugging
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    print("Loaded dataset\n")
    

    # Create an initial model with random values for each parameter (value ranges are from original code)
    learning_rate = random.uniform(0.0001, 0.009)
    dropout_rate = random.uniform(0.1, 0.5)
    dim_feedforward = random.randint(8, 128)
    batch = random.choice(range(16, 129, 16))
    encoder_layers = random.randint(1, 3)
    decoder_layers = random.randint(1, 3)
    activation_function = random.randint(1, 4) #1=relu, 2=leaky_relu, 3=tanh, 4=sigmoid
    time_warping = random.randint(0, 1) #0=False, 1=True
    time_masking = random.randint(0, 1) #0=False, 1=True
    gaussian_noise = 10 ** (random.uniform(math.log10(1e-4), math.log10(1e-1)))
    use_linear_embedding = random.randint(0, 1) #0=don't use, 1=use linear embedding

    if config.dataset in ['MSL', 'SMAP', 'SWaT', 'WADI', 'SMD', 'NAB', 'MBA', 'UCR']:
        phase_type = random.randint(1, 2) #1=2phase, 2=iterative
    else:
        phase_type = random.randint(1, 3) #1=1phase, 2=2phases, 3=iterative
    
    self_conditioning = random.randint(0, 1) #0=False, 1=True
    layer_norm = random.randint(0, 1) #0=False, 1=True
    positional_encoding_type = random.randint(1, 2) #1=sinusoidal, 2=fourier
    num_ffn_layers = random.randint(1, 3)
    nhead = labels.shape[1]

    init_params = [learning_rate, dropout_rate, dim_feedforward, batch, encoder_layers, decoder_layers, activation_function,time_warping, time_masking, gaussian_noise, use_linear_embedding, phase_type,self_conditioning, layer_norm, positional_encoding_type, num_ffn_layers, nhead]
    print("Created initial model: ", init_params, "\n")


    # Run CMA-ES optimization with initial solutions and step size
    sigma_0 = 0.5
    es = cma.CMAEvolutionStrategy(init_params, sigma_0)


    # Optimize using above objective function, each solution is a vector of param values
    # while not es.stop():
    for i in range(1, 5):
        print("\nGENERATION: ", i, "\n" )
        solutions = es.ask()
        fitnesses = []
        print("Got some solutions: ", solutions, "\n")

        # Finding fitness of each model
        for model in solutions:
            print("Model suggestion: ", model)
            fitness = fitness_function(model)
            print("Its fitness: ", fitness, "\n")
            fitnesses.append(fitness)
        
        # Getting feedback on the model
        es.tell(solutions, fitnesses)
        
        # Log
        es.logger.add()
        es.disp()


    # Best result
    best_solution = es.result.xbest
    best_fitness = - fitness_function(best_solution)
    
    print("\nBest solution found:")
    print(f"x = {best_solution}")
    print(f"Fitness = {best_fitness:.4f}")




if __name__ == '__main__':
    # Start tracking power by running the track_power.py script first in a separate terminal
    # Wait for some values to populate the CSV file then proceed here
    main()