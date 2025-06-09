from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib
import os
from collections import defaultdict
import re
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import resample


from openpyxl import load_workbook
import numpy as np


import time
from datetime import datetime
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from collections import Counter
import pickle


from  scipy.signal.windows import hann
import numpy as np
import math
from scipy.fftpack import fft,ifft

import os
import numpy as np
import math
import scipy.io as sio

from openpyxl import load_workbook

from pykalman import KalmanFilter

from scipy.ndimage import uniform_filter1d

from helper_functions import *
from ml_functions import *


#Directories
#data_dir = r"/content/drive/MyDrive/Tesis_code/Adultos/Data/de_mov_avg"
#results_dir = '/content/drive/MyDrive/Tesis_code/Adultos/Results/ML_v3(intra_subject)_avg_DE'
data_dir = r"C:\Users\Alonso\Desktop\Magister\train_educa\datasets\de_mov_avg"
results_dir = r'C:\Users\Alonso\Desktop\Magister\train_educa\Results\ML_v3(intra_subject)_avg_DE'

os.makedirs(results_dir, exist_ok=True)

data_files = [
    f for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f)) and f.endswith(".npz")
]

labels = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

model_keys = [
    'LogisticRegression',
    #'DecisionTreeClassifier',
    #'RandomForestClassifier',
    #'GradientBoostingClassifier',
    #'XGBClassifier',
    #'MLPClassifier',
    #'SVC',
    #'SGDClassifier'
]

est_params_dict = get_est_params_dict(model_keys)

#Training loop
for idx, file in enumerate(data_files):

  print(f"Sesion {idx}, {file}")
  path = os.path.join(data_dir, file)
  loaded = np.load(path, allow_pickle=True)

  X_train = loaded['subject_data'][0:9]
  y_train = np.array(labels[0:9])
  X_test = loaded['subject_data'][9:15]
  y_test = np.array(labels[9:15])

  X_train, X_test, y_train, y_test = generate_datasets(X_train, X_test, y_train, y_test)

  #print(X_train.shape, X_test.shape)
  #print(y_train.shape, y_test.shape)

  #Count train labels from each class
  count_labels(y_train)
  count_labels(y_test)

  # 4. Ejecutar entrenamiento, validación y test con tus funciones
  results_val, results_test, cm_test, models_dicc = run_short_version(est_params_dict, X_train, X_test, y_train, y_test, sample_weight_On = None, SMOTE_on= None, RandomUnderSampler_on = None)

  # 5. Mostrar resultados
  #print("\n🔍 Validación:")
  #for model, metrics in results_val.items():
  #    print(f"{model}: {metrics}")

  print("\n🧪 Test:")
  for model, metrics in results_test.items():
      print(f"{model}: {metrics}")

  for model, cm in cm_test.items():
      print(f"\n🔍 Confusion Matrix for {model}:")
      # Crear DataFrame con etiquetas
      cm_df = pd.DataFrame(cm,
                          index=["Actual 0", "Actual 1", "Actual 2"],
                          columns=["Predicted 0", "Predicted 1", "Predicted 2"])
      print(cm_df)

  #Save subject results
  session_results = {
      'results_val': results_val,
      'results_test': results_test,
      'cm_test': cm_test,
      'models_dicc': models_dicc
  }

  filename = file.replace(".npz", ".plk")
  with open(os.path.join(results_dir, filename), 'wb') as f:
      pickle.dump(session_results, f)
