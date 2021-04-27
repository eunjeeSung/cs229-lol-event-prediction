import configparser
import json
import requests
import time
import os
import math

import pandas as pd
import numpy as np
from tqdm import tqdm

cfg = configparser.ConfigParser()
cfg.read('../config.cfg')


def aggregate_rmtpp():
    def f_team_event_(df):
        return field_to_val_(EVENT_DICT, df.type, df.killerId)

    def f_detailed_team_event_(df):
        if isinstance(df.monsterType, str):
            return field_to_val_(DETAILED_EVENT_DICT, df.monsterType, df.killerId)
        elif isinstance(df.laneType, str):
            return field_to_val_(DETAILED_EVENT_DICT, df.laneType, df.killerId)
        elif isinstance(df.buildingType, str):
            return field_to_val_(DETAILED_EVENT_DICT, df.buildingType, df.killerId)
        else:
            return field_to_val_(DETAILED_EVENT_DICT, df.type, df.killerId)

    def field_to_val_(table, key, killer_id):
        team_buf = 0 if (1 <= killer_id <= 5) else len(table)
        if key in table:
            return table[key] + team_buf
        else:
            return None

    OBJECT_COLS = ["ELITE_MONSTER_KILL", "BUILDING_KILL"]
    CHAMP_COLS = ["CHAMPION_KILL"]

    EVENT_DICT = {"WARD_KILL": 0,
                       "CHAMPION_KILL": 1,
                       "ELITE_MONSTER_KILL": 2,
                       "BUILDING_KILL": 3}

    DETAILED_EVENT_DICT = {"WARD_KILL": 0,
                            "CHAMPION_KILL": 1,
                            "DRAGON": 2,
                            "RIFTHERALD": 3,
                            "BARON_NASHOR": 4,
                            "MID_LANE": 5,
                            "TOP_LANE": 6,
                            "BOT_LANE": 7,
                            "INHIBITOR_BUILDING": 8}

    events_reader = pd.read_csv('events_full.csv',
                                encoding='cp949',
                                iterator=True,
                                chunksize=1000)

    for i, chunk in enumerate(tqdm(events_reader)):
        game_ids = chunk['gameId']
        unique_game_ids = list(set(game_ids))

        for gid in game_ids:
            game_chunk = chunk[chunk['gameId'] == gid]

            game_chunk['event_marker'] = game_chunk['type'].map(EVENT_DICT)
            game_chunk['team_event_marker'] = \
                game_chunk.apply(lambda x: f_team_event_(x), axis=1)
            game_chunk['detailed_team_event_marker'] = \
                game_chunk.apply(lambda x: f_detailed_team_event_(x), axis=1)
            game_chunk['timestamp'] /= 1000

            game_chunk = game_chunk[~game_chunk['detailed_team_event_marker'].isnull()]        
            game_chunk = game_chunk[['timestamp', 'type',
                                    'event_marker', 'team_event_marker', 'detailed_team_event_marker']]    
            game_chunk = game_chunk.sort_values(by=['timestamp'])

            if os.path.exists('events/{game_id}.csv'):
                game_chunk.to_csv(f'events/{gid}.csv', mode='a', header=False, index=False, encoding='cp949')
            else:
                game_chunk.to_csv(f'events/{gid}.csv', index=False, encoding='cp949')

def get_games_list():
    gfiles = [f for f in os.listdir('events') \
                if os.path.isfile(os.path.join('events', f)) and os.stat(os.path.join('events', f)).st_size > 200]
    gids = [file_name.split('.')[0] for file_name in gfiles]
    df = pd.DataFrame()
    df['gfiles'], df['gids'] = gfiles, gids
    df.to_csv('games_list.csv', encoding='cp949')
    return gids, gfiles


def clean_data():
    gfiles = [f for f in os.listdir('events') \
                if os.path.isfile(os.path.join('events', f)) and os.stat(os.path.join('events', f)).st_size > 200]
    gids = [file_name.split('.')[0] for file_name in gfiles]

    for gfile, gid in zip(gfiles, gids):
        df = pd.read_csv(os.path.join('events', gfile))
        df = df.drop_duplicates()
        try:
            df = df.sort_values(['timestamp'])
        except Exception as err:
            print(f'{gid}: {err}')
        df.to_csv(f'events/{gid}.csv', index=False, encoding='cp949')

def check_columns():
    df = pd.read_csv('./games_list.csv')
    for gfile in df['gfiles']:
        game_df = pd.read_csv(os.path.join('./events', gfile))
        try:
            timestamp, marker = game_df['timestamp'], game_df['event_type']
        except Exception as err:
            print(gfile)

def check_null():
    df = pd.read_csv('./games_list.csv')
    for gfile in df['gfiles']:
        game_df = pd.read_csv(os.path.join('./events', gfile))
        if sum(game_df.isnull().sum()) != 0:
            print(gfile)

    
if __name__ == "__main__":
    aggregate_rmtpp()
    # get_games_list()
    #clean_data()
    # check_columns()
    #check_null()