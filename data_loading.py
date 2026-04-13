"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
import os

def air_compressor_loader(folder_path, seq_len):
  """
  Loads .dat files from the Healthy folder
  Windows them for TimeGAN
  """
  combined_data = []

  # Get list of all .dat files
  file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.dat')])
  print(f"Loading {len(file_list)} files from {folder_path}")

  for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    data = np.loadtxt(file_path, delimiter=',')
    combined_data.append(data)
  
  # Flatten all 225 recordings into a long 1D array, then reshape
  flat_data = np.concatenate(combined_data).reshape(-1, 1)

  # Normalization (TimeGAN assumes data is normalized between 0 and 1)
  # Use MinMaxScaler function
  norm_data = MinMaxScaler(flat_data)

  # Windowing (Slice into sequences)
  temp_data = []
  for i in range(0, len(norm_data) - seq_len):
    _x = norm_data[i:i + seq_len]
    temp_data.append(_x)

  # Shuffle the sequences
  idx = np.random.permutation(len(temp_data))
  output_data = []
  for i in range(len(temp_data)):
    output_data.append(temp_data[idx[i]])
  
  return output_data


def MinMaxScaler(data):
  """Min Max normalizer.
  Normalize data between 0 and 1
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """

  # Min-Max Normalization = (x - min(x)) / (max(x) - min(x))

  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  # Add 1e-7 to avoid zero division (case where min = max, and denominator = 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data

def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no): # Generate no samples
    # Initialize each time-series - each individual sample
    temp = list()
    # For each feature - # of sine waves
    for k in range(dim):
      # Randomly drawn frequency and phase
      # Each sine wave in each sample has different frequency and phase - creates diversity in the data
      freq = np.random.uniform(0, 0.1) # Frequency - how fast the wave repeats
      phase = np.random.uniform(0, 0.1) # Phase - where it starts (horizontal shift)
      
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
    
    # Align row/column - from dim x seq_len to seq_len x dim
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1] - sine is naturally between -1 and 1
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','energy']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
        
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data