"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from datetime import datetime

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation, air_compressor_loader
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

class Logger(object):
    """Redirects stdout to both the console and a log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  # Experiment Directory
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  run_name = f'run_{args.data_name}_{timestamp}'
  exp_dir = os.path.join('./experiments', run_name)
  
  for sub in ['ckpt', 'npy', 'plots', 'logs']:
    os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    
  log_path = os.path.join(exp_dir, 'logs', 'training.log')
  sys.stdout = Logger(log_path)
  
  print(f'Starting Experiment: {run_name}')
  print(f'All outputs will be saved to: {exp_dir}')
  print(f'Argumnts: {args}\n')
  
  
  ## Data loading
  if args.data_name in ['stock', 'energy']:
    ori_data = real_data_loading(args.data_name, args.seq_len)
  elif args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
  elif args.data_name == 'air_healthy':
    print("Loading Air Compressor Dataset")
    ori_data = air_compressor_loader('./AirCompressor_Data/Healthy', args.seq_len)
    
  print(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
  # parameters['exp_dir'] = exp_dir
  
  ## DELETE LATER
  if args.use_saved_data:
    parameters['ckpt_path'] = "./experiments/run_air_healthy_20260413_212624/ckpt/final_model.ckpt"
      
  generated_data = timegan(ori_data, parameters)
  np.save(os.path.join(exp_dir, "npy", "generated_data.npy"), generated_data)
  np.save(os.path.join(exp_dir, "npy", "ori_data.npy"), ori_data)
  print("Saved generated data.")  
  print('Finish Synthetic Data Generation')
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. Discriminative Score
  print('Computing discriminative score...')
  discriminative_score = list()
  for i in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)
    print(f'Iteration {i+1}: {temp_disc}')
      
  metric_results['discriminative'] = np.mean(discriminative_score)
      
  # 2. Predictive score
  predictive_score = list()
  for i in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)
    print(f"  Iteration {i+1}: {temp_pred}")
      
  metric_results['predictive'] = np.mean(predictive_score)     
        
  # 3. Visualization (PCA and tSNE)
  print("Generating Visualizations")
  visualization(ori_data, generated_data, 'pca', os.path.join(exp_dir, 'plots', 'pca_final.png'))
  visualization(ori_data, generated_data, 'tsne', os.path.join(exp_dir, 'plots', 'tsne_final.png'))

  print(f"\nFinal Metric Results: {metric_results}")
  print(f"Experiment {run_name} complete.")

  return ori_data, generated_data, metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['sine','stock','energy','air_healthy'],
      default='air_healthy',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=50000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  parser.add_argument(
      '--use_saved_data',
      help='Skip training and load saved npy data',
      action='store_true'
  )
  
  args = parser.parse_args() 
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)