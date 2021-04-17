import configparser
import json
import requests
import time

import os
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
from tqdm import tqdm

cfg = configparser.ConfigParser()
cfg.read('../config.cfg')

def split_by_game():
    events_reader = pd.read_csv(cfg['file']['events_interval_path'],
                                encoding='cp949',
                                iterator=True,
                                chunksize=1000)                        

    for i, chunk in enumerate(tqdm(events_reader)):
        game_ids = chunk['gameId']
        unique_game_ids = list(set(game_ids))

        for gid in unique_game_ids:
            try:
                game_chunk = chunk[chunk['gameId'] == gid]

                start_time, end_time = game_chunk['tick'].min(), game_chunk['tick'].max()
                if not os.path.exists(f'events/{gid}.csv'):
                    timestamps = range(0, end_time+1)
                else:
                    timestamps = range(start_time, end_time+1)                
                game_chunk = game_chunk.set_index('tick').reindex(timestamps)
                
                empty_row_idxs = game_chunk['gameId'].isna()
                game_id = game_chunk['gameId'][game_chunk['gameId'].first_valid_index()]
                vals = [game_id,0.0,0.0,0.0,0.0,0.0,0.0,0.0,[],0.0,0.0,0.0,0.0,0.0,0.0,[],0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,[],0.0,0.0,0.0,0.0,0.0,0.0,[],0.0,0.0]
                game_chunk.at[empty_row_idxs] = np.expand_dims(np.array(vals, dtype=object), axis=0)

                if not os.path.exists(f'events/{gid}.csv'):
                    #print('Creating a new file...')
                    game_chunk.to_csv(f'events/{gid}.csv', index=True, encoding='cp949')
                else:
                    game_chunk.to_csv(f'events/{gid}.csv', mode='a', header=False, index=True, encoding='cp949')
            except Exception as err:
                print(f"{err}: gid {gid}")    

def remove_duplicates_from_games():
    gids, gfiles = get_games_list()

    for gid, gfile in zip(gids, gfiles):
        df = pd.read_csv(f'events/{gfile}', encoding='cp949')
        bf_len = len(df)
        df = df.drop_duplicates()
        af_len = len(df)  
        if bf_len != af_len:
            print(f'{gid}: {bf_len} -> {af_len}')
        df.to_csv(f'events_cleaned/{gid}.csv', index=False, encoding='cp949')              

def get_games_list():
    gfiles = [f for f in listdir('events') if isfile(join('events', f))]
    gids = [file_name.split('.')[0] for file_name in gfiles]
    df = pd.DataFrame()
    df['gfiles'], df['gids'] = gfiles, gids
    df.to_csv('games_list.csv', encoding='cp949')
    return gids, gfiles

if __name__ == "__main__":
    remove_duplicates_from_games()