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


def aggregate_events_by_interval(ms_interval):
    USED_COLUMNS = ['gameId', 'tick' \
    ,'blueKill','blueDeath','blueAssist'\
    ,'blueWardPlaced','blueWardKills','blueFirstTower','blueFirstInhibitor','blueFirstTowerLane'\
    ,'blueTowerKills','blueMidTowerKills','blueTopTowerKills','blueBotTowerKills'\
    ,'blueInhibitor','blueFirstDragon','blueDragonType','blueDragon','blueRiftHeralds'\
    ,'redKill','redDeath','redAssist'\
    ,'redWardPlaced','redWardKills','redFirstTower','redFirstInhibitor'\
    ,'redFirstTowerLane'\
    ,'redTowerKills','redMidTowerKills','redTopTowerKills','redBotTowerKills'\
    ,'redInhibitor','redFirstDragon','redDragonType','redDragon','redRiftHeralds']

    events_reader = pd.read_csv(cfg['file']['events_cleaned_info_path'],
                                encoding='cp949',
                                iterator=True,
                                chunksize=3000)
    curr_tick = 0

    for i, chunk in enumerate(tqdm(events_reader)):
        # code copied from https://shinminyong.tistory.com/30
        chunk_data = []

        for j, event in chunk.iterrows():
            tick = event['timestamp'] // ms_interval
            timestamp = event['timestamp']
            if (i == 0) or (tick != curr_tick):
                if (not (i == 0 and j==0))  and (tick != curr_tick):
                    chunk_data.append([game_id, curr_tick \
                        ,blue_kill,blue_death,blue_assist,blue_wardplace,blue_wardkill\
                        ,blue_firsttower,blue_firstinhibitor,blue_firsttowerlane,blue_tower\
                        ,blue_midtower,blue_toptower,blue_bottower,blue_inhibitor,blue_firstdragon\
                        ,blue_dragontype,blue_dragon,blue_rift\
                        ,red_kill,red_death,red_assist,red_wardplace,red_wardkill\
                        ,red_firsttower,red_firstinhibitor,red_firsttowerlane,red_tower\
                        ,red_midtower,red_toptower,red_bottower,red_inhibitor,red_firstdragon\
                        ,red_dragontype,red_dragon,red_rift])

                curr_tick = tick

                game_id = event['gameId']
                blue_kill, red_kill = 0,0
                blue_firstkill, red_firstkill = 0,0
                blue_assist, red_assist = 0,0
                red_death, blue_death = 0,0
                blue_wardplace, red_wardplace = 0,0
                blue_wardkill, red_wardkill = 0,0
                blue_elite, red_elite = 0,0
                blue_rift, red_rift = 0,0
                blue_dragon, red_dragon = 0,0
                blue_baron, red_baron = 0,0
                blue_firstdragon, red_firstdragon = 0,0
                blue_dragontype, red_dragontype = [],[]
                blue_firstbaron, red_firstbaron = 0,0
                blue_tower,red_tower = 0,0
                blue_firsttower, red_firsttower = 0,0
                blue_firsttowerlane, red_firsttowerlane = [],[]
                blue_midtower, red_midtower = 0,0
                blue_toptower, red_toptower = 0,0
                blue_bottower, red_bottower = 0,0
                blue_inhibitor, red_inhibitor = 0,0
                blue_firstinhibitor, red_firstinhibitor = 0,0           

            if event['type'] == 'WARD_KILL':
                if 1 <= event['killerId'] <= 5:
                    blue_wardkill += 1
                else:
                    red_wardkill += 1

            elif event['type'] == 'WARD_PLACED':
                if 1 <= event['creatorId'] <= 5:
                    blue_wardplace += 1
                else:
                    red_wardplace += 1

            elif event['type'] == 'CHAMPION_KILL': 
                if 1 <= event['killerId'] <= 5:
                    if red_kill ==0 and blue_kill ==0:
                        blue_firstkill += 0
                    blue_kill += 1
                    blue_assist += len(event['assistingParticipantIds'])
                    red_death += 1
                else:
                    if red_kill ==0 and blue_kill ==0:
                        red_firstkill += 0
                    red_kill += 1
                    red_assist += len(event['assistingParticipantIds'])
                    blue_death += 1

            elif event['type'] == 'ELITE_MONSTER_KILL':
                if 1 <= event['killerId'] <= 5:
                    blue_elite += 1
                    if event['monsterType']== 'DRAGON':
                        if red_dragon ==0 and blue_dragon == 0:
                                blue_firstdragon += 1
                        #blue_dragontype.append(event['monsterSubType'])
                        blue_dragon += 1
                    elif event['monsterType']== 'RIFTHERALD':
                        blue_rift += 1
                    elif event['monsterType']== 'BARON_NASHOR':
                        if red_baron ==0 and blue_dragon == 0:
                                blue_firstbaron += 1
                        else:
                            pass
                        blue_baron += 1
                else:
                    red_elite += 1
                    if event['monsterType']== 'DRAGON':
                        if red_dragon ==0 and blue_dragon == 0:
                                red_firstdragon += 1
                        #red_dragontype.append(event['monsterSubType'])
                        red_dragon += 1
                    elif event['monsterType']== 'RIFTHERALD':
                        red_rift += 1
                    elif event['monsterType']== 'BARON_NASHOR':
                        if red_baron ==0 and blue_dragon == 0:
                                red_firstbaron += 1
                        red_baron += 1

            elif event['type'] == 'BUILDING_KILL':
                if 1 <= event['killerId'] <= 5:
                    if event['buildingType'] == 'TOWER_BUILDING':
                        if red_tower == 0 and blue_tower ==0:
                            blue_firsttower += 1
                            blue_firsttowerlane.append(event['laneType'])
                        blue_tower += 1
                        if event['laneType'] == 'MID_LANE':
                            blue_midtower += 1
                        elif event['laneType'] == 'TOP_LANE':
                            blue_toptower += 1
                        elif event['laneType'] == 'BOT_LANE':
                            blue_bottower += 1   
                    elif event['buildingType'] == 'INHIBITOR_BUILDING':
                        if red_inhibitor == 0 and blue_inhibitor ==0:
                            blue_firstinhibitor += 1
                        blue_inhibitor += 1
                else:
                    if event['buildingType'] == 'TOWER_BUILDING':
                        if red_tower == 0 and blue_tower ==0:
                            red_firsttower += 1
                            red_firsttowerlane.append(event['laneType'])
                        red_tower += 1

                        if event['laneType'] == 'MID_LANE':
                            red_midtower += 1
                        elif event['laneType'] == 'TOP_LANE':
                            red_toptower += 1
                        elif event['laneType'] == 'BOT_LANE':
                            red_bottower += 1
                    elif event['buildingType'] == 'INHIBITOR_BUILDING':
                        if red_inhibitor == 0 and blue_inhibitor ==0:
                            red_firstinhibitor += 1
                        red_inhibitor += 1

        if len(chunk_data) != 0:
            chunk_data = pd.DataFrame(chunk_data)
            chunk_data.columns = USED_COLUMNS
            if i==0:
                print('Creating a new file...')
                chunk_data.to_csv('events_interval_60.csv', index=False, encoding='cp949')
            else:
                chunk_data.to_csv('events_interval_60.csv', mode='a', header=False, index=False, encoding='cp949')

    
if __name__ == "__main__":
    aggregate_events_by_interval(60000)