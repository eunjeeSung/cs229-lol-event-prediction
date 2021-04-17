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
API_KEY = cfg['riot-api']['api-key']


def get_summoner_id(name):
    sohwan = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/" + name +'?api_key=' + API_KEY
    r = requests.get(sohwan)
    return r.json()['id']

def get_summoner_info(id):
    tier_url = "https://kr.api.riotgames.com/lol/league/v4/entries/by-summoner/" + id +'?api_key=' + API_KEY
    r2  = requests.get(tier_url)
    return r2.json()

def get_grandmaster_info():
    grandmaster = 'https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5?api_key=' + API_KEY
    r = requests.get(grandmaster)#그마데이터 호출
    league_df = pd.DataFrame(r.json())

    league_df.reset_index(inplace=True)#수집한 그마데이터 index정리
    league_entries_df = pd.DataFrame(dict(league_df['entries'])).T #dict구조로 되어 있는 entries컬럼 풀어주기
    league_df = pd.concat([league_df, league_entries_df], axis=1) #열끼리 결합

    league_df = league_df.drop(['index', 'queue', 'name', 'leagueId', 'entries', 'rank'], axis=1)
    league_df.info()
    league_df.to_csv('그마데이터.csv', index=False, encoding = 'cp949')#중간저장
    return league_df

def get_grand_master_ids(league_df):
    if 'accountId' not in league_df:
        league_df['accountId'] = ''

    print("Collecting grandmaster summoners data...")
    for i in tqdm(range(len(league_df))):
        try:
            sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + API_KEY 
            r = requests.get(sohwan)
        
            if r.status_code == 404:
                continue
            while r.status_code == 429:
                time.sleep(5)
                sohwan = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + league_df['summonerName'].iloc[i] + '?api_key=' + API_KEY 
                r = requests.get(sohwan)
            
            account_id = r.json()['accountId']
            league_df.iloc[i, -1] = account_id
        except Exception as e:
            print(f'{i}-th iteration: {e}')
            raise

    league_df = league_df.dropna()
    league_df.info()
    league_df.to_csv('gma_ids.csv', index=False, encoding = 'cp949')  
    return league_df

def get_game_ids(league_df):
    match_info_df = pd.DataFrame()
    season = str(13)
    print("Collecting match data...")
    for i in tqdm(range(len(league_df))):
        try:
            match0 = 'https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/' + league_df['accountId'].iloc[i] +'?season=' + season + '&api_key=' + API_KEY
            r = requests.get(match0)

            if r.status_code == 404:
                continue
            
            while r.status_code == 429:
                time.sleep(5)
                match0 = 'https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/' + league_df['accountId'].iloc[i]  +'?season=' + season + '&api_key=' + API_KEY
                r = requests.get(match0)

            match_info_df = pd.concat([match_info_df, pd.DataFrame(r.json()['matches'])])
        except Exception as e:
            print(f'{i}-th iteration: {e}')
            raise
    
    match_info_df.dropna()
    match_info_df.info()
    match_info_df.to_csv('match_ids.csv', index=False, encoding = 'cp949')
    return match_info_df

def get_game_info(game):
    match_url = "https://kr.api.riotgames.com/lol/match/v4/timelines/by-match/{}?api_key=" + API_KEY

    match_url = \
        'https://kr.api.riotgames.com/lol/match/v4/timelines/by-match/{}?api_key=' +  API_KEY
    start = 0

    for b in tqdm(range(start, len(game))):
        try:
            game_id = game['gameId'].iloc[b]
            req = requests.get(match_url.format(game_id))

            # HTTP response 확인 및 API 쿼터 모두 사용 시 대기
            req = _check_status(req, b, match_url, game_id)
            
            # json data에서 필요한 frames 필드만
            if 'frames' not in req.json:                                               
                print(req)
                continue
            frames = req.json()['frames']

            # 타임스탬프 별 참가자 정보, 이벤트 정보 추출
            participants_info_df = None
            #participants_info_df = _get_participants_timestamp_info(game_id, frames)
            events_info_df = _get_events_timestamp_info(game_id, frames)

            # 파일에 추가
            if b==0:
                print('Creating a new file...')
                #participants_info_df.to_csv('participants_full.csv', index=False, encoding = 'cp949')
                events_info_df.to_csv('events_full.csv', index=False, encoding = 'cp949')
            else:
                #participants_info_df.to_csv('participants_full.csv', mode='a', header=False, index=False, encoding = 'cp949')
                events_info_df.to_csv('events_full.csv', mode='a', header=False, index=False, encoding = 'cp949')
        
        except Exception as e:
            print(e)
            raise

    print('data crawling success')
    return participants_info_df, events_info_df

def _get_participants_timestamp_info(game_id, frames):
    data = []
    for frame in frames:
        participant_frames = frame['participantFrames']
        timestamp = frame['timestamp']

        for _, part in participant_frames.items():
            part['gameId'] = game_id
            part['timestamp'] = timestamp            
            part['posX'] = part['position']['x'] if 'position' in part else None
            part['posY'] = part['position']['y'] if 'position' in part else None
            data.append(part)

    columns = list(frames[0]['participantFrames']['1'].keys())
    participants_df = pd.DataFrame(data)
    participants_df.columns = columns
    participants_df = participants_df.drop(columns=['position'])
    return participants_df

def _get_events_timestamp_info(game_id, frames):
    def _get_field(d, field):
        return d[field] if field in d else None

    data = []
    for frame in frames:
        event_frames = frame['events']

        for event in event_frames:
            data_row = {}
            data_row['gameId'] = game_id            
            data_row['type'] = _get_field(event, 'type')
            data_row['timestamp'] = _get_field(event, 'timestamp')
            data_row['participantId'] = _get_field(event, 'participantId')
            data_row['itemId'] = _get_field(event, 'itemId')
            data_row['posX'] = event['position']['x'] if 'position' in event else None
            data_row['posY'] = event['position']['y'] if 'position' in event else None
            data_row['killerId'] = _get_field(event, 'killerId')
            data_row['assistingParticipantIds'] = _get_field(event, 'assistingParticipantIds')
            data_row['teamId'] = _get_field(event, 'teamId')
            data_row['buildingType'] = _get_field(event, 'buildingType')
            data_row['laneType'] = _get_field(event, 'laneType')
            data_row['towerType'] = _get_field(event, 'towerType')
            data_row['creatorId'] = _get_field(event, 'creatorId')
            data_row['monsterType'] = _get_field(event, 'monsterType')
            data.append(data_row)
    columns = ['gameId', 'type', 'timestamp', 'participantId', 'itemId',
                'posX', 'posY', 'killerId', 'assistingParticipantIds', 'teamId',
                'buildingType', 'laneType', 'towerType',
                'creatorId', 'monsterType']
    events_df = pd.DataFrame(data)
    events_df.columns = columns
    return events_df

def _check_status(req, b, match_url, game_id):
    if req.status_code == 200:
        pass

    elif req.status_code == 429:
        print('api cost full : infinite loop start')
        print('loop location : ',b)
        start_time = time.time()
        while True:
            if req.status_code == 429:
                print('try 10 second wait time')
                time.sleep(10)
                req = requests.get(match_url.format(game_id))
                print(req.status_code)
            elif req.status_code == 200:
                print('total wait time : ', time.time() - start_time)
                print('recovery api cost')
                break
            else:
                print(req.status_code)

    elif req.status_code == 503:
        print('service available error')
        start_time = time.time()
        while True:
            if req.status_code == 503 or req.status_code == 429:
                print('try 10 second wait time')
                time.sleep(10)
                req = requests.get(match_url.format(game_id))
                print(req.status_code)
            elif req.status_code == 200:
                print('total error wait time : ', time.time() - start_time)
                print('recovery api cost')
                break
            else:
                print(req.status_code)

    elif req.status_code == 403: # api갱신이 필요
        print('you need api renewal')
        print('break')
    
    return req

def cleanse_participants(reader):
    for i, chunk in enumerate(tqdm(reader)):
        chunk_cleaned = chunk.drop_duplicates()
        if len(chunk) != len(chunk_cleaned):
            print(f"duplicates found: {i} / # duplicates: {len(chunk) - len(chunk_cleaned)}")

        #파일에 추가
        if i==0:
            print('Creating a new file...')
            chunk.to_csv('participants_cleaned.csv', index=False, encoding='cp949')
        else:
            chunk.to_csv('participants_cleaned.csv', mode='a', header=False, index=False, encoding='cp949')


if __name__ == "__main__":
    # s_name = "hide on bush"
    # s_id = get_summoner_id(s_name)
    # s_info = get_summoner_info(s_id)

    #league_df = get_grandmaster_info()
    #league_df = get_grand_master_ids(league_df)

    #file_path = cfg['file']['gma_ids_path']
    #league_df = pd.read_csv(file_path, encoding='cp949')    
    #match_info_df = get_game_ids(league_df)

    # match_info_df = pd.read_csv(cfg['file']['match_info_path'], encoding='cp949')
    # game_info_df = get_game_info(match_info_df)

    game_info_reader = pd.read_csv(cfg['file']['participants_info_path'], encoding='cp949',
                                iterator=True,
                                chunksize=1000)
    cleanse_participants(game_info_reader)