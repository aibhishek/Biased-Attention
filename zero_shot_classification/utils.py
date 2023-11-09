import pandas as pd
import numpy as np
import glob
from collections import OrderedDict
from operator import itemgetter
from .zero_shot_classifier import *

keywords = ['cleaner', 'statistician', 'midwife', 'auctioneer', 'photographer', 'geologist', 'athlete', 'cashier',
'dancer', 'housekeeper', 'accountant', 'physicist', 'gardener', 'dentist', 'blacksmith', 'psychologist', 'supervisor',
'mathematician', 'surveyor', 'tailor', 'designer', 'economist', 'mechanic', 'laborer', 'postmaster', 'broker', 'chemist', 'librarian',
'attendant', 'clerk', 'musician', 'porter', 'scientist', 'carpenter', 'sailor', 'instructor', 'sheriff', 'pilot', 'inspector', 'construction worker',
'baker', 'administrator', 'architect', 'collector', 'operator', 'surgeon', 'driver', 'painter', 'conductor', 'nurse', 'chef', 'engineer',
'salesperson', 'lawyer', 'clergy', 'physician', 'farmer', 'manager', 'guard', 'artist', 'official', 'police', 'doctor',
'professor', 'student', 'judge', 'teacher', 'author', 'secretary', 'soldier', 'chief executive officer', 'product owner', 'fitter', 'welder', 'hairdresser', 'beautician',
'lifeguard', 'journalist', 'firefighter', 'customer service executive', 'support worker', 'youtuber' ,'machine operator', 'plumber', 'electrician', 'jewellery maker',
 'digital content creator', 'programmer', 'retail assistant' , 'trainer' , 'food server' , 'business analyst', 'warehouse operative', 'coach', 'handyman', 'psychiatrist', 'counsellor' ,'supply chain associate', 'investment banker', 'filmmaker']

model_ls = ['RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'ViT-B/32']

def run_classifier():
    for model_name in model_ls:
        print('-'*50)
        print('Zero-shot predictions using '+model_name)
        print('-'*50)
        get_zero_shot_classifications(model_name, './images/', keywords)

    


def generate_df(df_path):
    df = pd.read_csv(df_path)
    key, values = np.unique(df['1st Label Man'],  return_counts=True)
    adj_cnts_man = {k:v for k,v in zip(key, values)}
    adj_cnts_man = OrderedDict(sorted(adj_cnts_man.items(), key=itemgetter(1),reverse=True))
    
    key, values = np.unique(df['1st Label Woman'],  return_counts=True)
    adj_cnts_woman = {k:v for k,v in zip(key, values)}
    adj_cnts_woman = OrderedDict(sorted(adj_cnts_woman.items(), key=itemgetter(1),reverse=True))
    
    return (pd.DataFrame(list(zip(list(adj_cnts_man.keys()), list(adj_cnts_man.values()), list(adj_cnts_woman.keys()), list(adj_cnts_woman.values()))),
                     columns=['Man', 'Man_Count', 'Woman', 'Woman_Count']))



