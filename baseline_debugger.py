import os
import sys
import pandas as pd
import torch
import numpy as np

from GraphTrafficLib.train.simple_baseline_trainer import SimpleBaselineTrainer

datafolder = '../datafolder/procdata'

data_path = f"{datafolder}/split_manhattan/full_year_lower_manhattan_2d.npy"
dropoff_data_path = None
weather_data_path = f"{datafolder}/LGA_weather_full_2019.csv"

baseline_trainer = SimpleBaselineTrainer()

baseline_trainer.load_data(data_path, weather_data_path)