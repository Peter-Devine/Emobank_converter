import re
import argparse
import os
import pandas as pd
import numpy as np

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped DailyDialog dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
parser.add_argument('--vad_bin_num', default="7", help='Number of bins to separate the VAD values into (if you edit this you need to also edit the BERT classifier code)')

args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output
VAD_BIN_NUM = int(args.vad_bin_num)

# Make the output directory if it does not currently exist
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

emobank = pd.read_csv(INPUT_PATH + "/emobank.csv")

def context_apply(current_id):
    position = re.finditer('_', current_id)

    if len(list(position)) > 1:
        position = re.finditer('_', current_id)
        
        position = [match.span() for match in position]
        
        position = [position[len(position)-2], position[len(position)-1]]
        
        stem_id = current_id[0:position[0][0]]
        start_index = int(current_id[position[0][0]+1:position[1][0]])
        
        context_mask = emobank["id"].str.match(stem_id+"_[0-9]+_"+str(start_index))
        context_mask_shifted = emobank["id"].str.match(stem_id+"_[0-9]+_"+str(start_index + 1))
        context_mask = context_mask | context_mask_shifted
        
        if context_mask.any():
            return emobank.loc[context_mask,:].iloc[0].text
        else:
            return ""
    else:
        return ""

context_list = list(map(context_apply, list(emobank["id"])))

emobank["context"] = context_list

fraction = 0.2

np.random.seed(seed=42)

test_indices = np.random.choice(emobank.index, size=int(round(fraction*emobank.shape[0])), replace=False)
train_indices = emobank.index.difference(test_indices)
dev_indices = np.random.choice(train_indices, size=int(round(fraction*len(train_indices))), replace=False)
train_indices = train_indices.difference(dev_indices)

emo_bank_train = emobank.loc[train_indices,:]
emo_bank_dev = emobank.loc[dev_indices,:]
emo_bank_test = emobank.loc[test_indices,:]

split_dataframe = {"train": emo_bank_train,
                  "dev": emo_bank_dev,
                  "test": emo_bank_test}

training_bins = {"V": "",
                "A": "",
                "D": ""}
    
for dataframe_type in ["train", "dev", "test"]:
    
    dataframe = split_dataframe[dataframe_type]
    
    emotion_columns = ["V", "A", "D"]
    
    for column in emotion_columns:
        number_of_bins = VAD_BIN_NUM
        bin_labels = [column+str(bin_label_index+1) for bin_label_index in range(number_of_bins)]
        if dataframe_type == "train":
            binned_data = pd.cut(dataframe[column], bins=number_of_bins, retbins=True)
            bins = binned_data[1]
            bins[0] = -999
            bins[len(bins)-1] = 999
            training_bins[column] = bins
        else:
            bins = training_bins[column]
        dataframe[column + "_binned"] = pd.cut(dataframe[column], bins=bins, labels=bin_labels)
     
    dataframe.reset_index(drop=True).to_csv(OUTPUT_PATH+"/"+dataframe_type+".tsv", sep='\t', encoding="utf-8")   
