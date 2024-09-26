import numpy as np
import csv
import matplotlib.pyplot as plt
import imutils
from court_detection import CourtDetector
from statistics_module import Statistics


heatmap = np.load('output/heatmapdata.npy', allow_pickle=True)
strokes = np.load('output/predictionsdata.npy', allow_pickle=True)


data_dict = strokes.item()

# Open a CSV file for writing
with open('output/strokeprediction.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['Video_Frame', 'Forehand_Probability', 'Backhand_Probability', 'Serve_Probability', 'Stroke'])
    
    # Write the data
    for key, value in data_dict.items():
        if isinstance(value, dict) and 'probs' in value and isinstance(value['probs'], np.ndarray):
            row = [key] + value['probs'].tolist() + [value['stroke']]
            writer.writerow(row)

np.savetxt("output/heatmapdata.csv", heatmap, delimiter=",")


bot_player = np.load('output/bottom_player.npy', mmap_mode='r')
np.savetxt("output/bottom_player_data.csv", bot_player, delimiter=",",header="X, Y", comments='')

top_player = np.load('output/top_player.npy', mmap_mode='r')
np.savetxt("output/top_player_data.csv", top_player, delimiter=",", header="X,Y", comments='')
