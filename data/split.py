import configparser
import json
import requests
import time
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

cfg = configparser.ConfigParser()
cfg.read('../config.cfg')


if __name__ == "__main__":
    events_reader = pd.read_csv(cfg['file']['events_interval_path'],
                                encoding='cp949',
                                iterator=True,
                                chunksize=1000)
    prev_game_id = ''

    for i, chunk in enumerate(tqdm(events_reader)):
        game_ids = chunk['gameId']
        unique_game_ids = list(set(game_ids))

        for gid in game_ids:
            game_chunk = chunk[chunk['gameId'] == gid]
            if os.path.exists('events/{game_id}.csv'):
                game_chunk.to_csv(f'events/{gid}.csv', mode='a', header=False, index=False, encoding='cp949')
            else:
                game_chunk.to_csv(f'events/{gid}.csv', index=False, encoding='cp949')