import re
from sklearn.model_selection import train_test_split
import argparse
import os
import pandas as pd

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

emotion_columns = ["V", "A", "D"]
for column in emotion_columns:
    number_of_bins = VAD_BIN_NUM
    bin_labels = [column+str(bin_label_index+1) for bin_label_index in range(number_of_bins)]
    binned_data = pd.cut(emobank[column], bins=number_of_bins, retbins=True)
    bins = binned_data[1]
    emobank[column + "_binned"] = pd.cut(emobank[column], bins=bins, labels=bin_labels)
    
emobank_train, emobank_test = train_test_split(emobank, test_size=0.3, random_state=42)

emobank_train, emobank_val = train_test_split(emobank_train, test_size=0.3, random_state=42)

emobank_train = emobank_train.reset_index(drop=True)
emobank_val = emobank_val.reset_index(drop=True)
emobank_test = emobank_test.reset_index(drop=True)

emobank_train.to_csv(OUTPUT_PATH+"/train.tsv", sep='\t', encoding="utf-8")
emobank_val.to_csv(OUTPUT_PATH+"/dev.tsv", sep='\t', encoding="utf-8")
emobank_test.to_csv(OUTPUT_PATH+"/test.tsv", sep='\t', encoding="utf-8")