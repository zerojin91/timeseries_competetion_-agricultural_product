# -*- coding: utf-8 -*-
"""lgbm_inference.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FAkZpe9-xQ-zBPqtioMaKCYpkfrepMs1
"""

import pandas as pd
import numpy as np
import lightgbm
from tqdm import tqdm
import warnings

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from sklearn.externals import joblib

import torch
import pandas as pd
import urllib.request
import json
import datetime
from datetime import timedelta
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from glob import glob

import multiprocessing
import pickle
import easydict

def fe(df, train=True):
    # 기존 전처리
    df_ = df.copy()

    for i in range(len(df_)):
        day= pd.to_datetime(df_.iloc[i]['date'])
        df_.iloc[i,1] = day.dayofweek 

    df_ = df_.fillna(0)     

    col_header = df_.columns.values.tolist()
    del(col_header[0:2])   

    df_[col_header] = df_[col_header].ewm(alpha = 0.5).mean()
 
    return df_

def preprocessing(temp_df, pum, len_lag) :
    # p_lag, q_lag 추가
    for lag in range(1,len_lag+1) :
      temp_df[f'p_lag_{lag}'] = -1
      temp_df[f'q_lag_{lag}'] = -1
      for index in range(lag, len(temp_df)) :
        temp_df.loc[index, f'p_lag_{lag}'] = temp_df[f'{pum}_가격(원/kg)'][index-lag] #1일전, 2일전, ... 가격을 feature로 추가
        temp_df.loc[index, f'q_lag_{lag}'] = temp_df[f'{pum}_거래량(kg)'][index-lag] #1일전, 2일전, ... 거래량을 feature로 추가
     
    # month 추가
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    temp_df['week'] = temp_df['date'].dt.week
    temp_df['month'] = temp_df['date'].dt.month
    temp_df['day'] = temp_df['date'].dt.day
 
    # 예측 대상(1w,2w,4w) 추가
    for week in ['1_week','2_week','4_week'] :
      temp_df[week] = 0
      n_week = int(week[0])
      for index in range(len(temp_df)) :
        try : temp_df[week][index] = temp_df[f'{pum}_가격(원/kg)'][index+7*n_week]
        except : continue

    # 불필요한 column 제거        
    temp_df = temp_df.drop(['date',f'{pum}_거래량(kg)',f'{pum}_가격(원/kg)'], axis=1)
    
    return temp_df

"""### metric 정의"""

def nmae(week_answer, week_submission):
    answer = week_answer.to_numpy()
    target_idx = np.where(answer!=0)
    true = answer[target_idx]
    pred = week_submission[target_idx]
    score = np.mean(np.abs(true-pred)/true)
    
    return score


def at_nmae(pred, dataset):
    y_true = dataset.get_label()
    week_1_answer = y_true[0::3]
    week_2_answer = y_true[1::3]
    week_4_answer = y_true[2::3]
    
    week_1_submission = pred[0::3]
    week_2_submission = pred[1::3]
    week_4_submission = pred[2::3]
    
    score1 = nmae(week_1_answer, week_1_submission)
    score2 = nmae(week_2_answer, week_2_submission)
    score4 = nmae(week_4_answer, week_4_submission)
    
    score = (score1+score2+score4)/3
    
    return 'score', score, False

### 학습 정의

def model_train(x_train, y_train, x_valid, y_valid) :
    params = {'learning_rate': 0.01, 
              'max_depth': 6, 
              'boosting': 'gbdt', 
              'objective': 'regression',  
              'is_training_metric': True, 
              'num_leaves': 100, 
              'feature_fraction': 0.8, 
              'bagging_fraction': 0.8, 
              'bagging_freq': 5, 
              'seed':42,
              'num_threads':8
             }

    model = lightgbm.train(params, 
                   train_set = lightgbm.Dataset(data = x_train, label = y_train),
                   num_boost_round = 10000, 
                   valid_sets = lightgbm.Dataset(data = x_valid, label = y_valid), 
                   init_model = None, 
                   early_stopping_rounds = 100,
                   feval = at_nmae,
                   verbose_eval = False
                    )
    
    return model

"""### 품목 및 품종별 모델 학습"""

def add_data(args):
    df_ = pd.read_csv(args.save_path+'all_df.csv')
    start_date = '2021-09-28'
    today = str(datetime.date.today().year)+str(datetime.date.today().month).zfill(2)+str(datetime.date.today().day).zfill(2)
    available_day = str(pd.to_datetime(today) - timedelta(days=1)).split(" ")[0].replace("-","")


    url = 'https://www.nongnet.or.kr/api/whlslDstrQr.do?sdate=' 
    data_list = []

    if start_date != today:
        print("데이터 수집 시작")
        for day in tqdm(list(pd.date_range(start_date, available_day, freq='D'))):
            day_t = str(day).split(" ")[0].replace("-","")
            response = urllib.request.urlopen(url+day_t).read()
            response = json.loads(response)
            data = pd.DataFrame(response['data'])

            if len(data) > 0 :
                data_list.append(data)
            else:
                continue
        
        target = [
          '배추', '무', '양파', '건고추', '마늘', '대파', '얼갈이배추', '양배추', 
          '깻잎', '시금치', '미나리', '당근', '파프리카', '새송이', '팽이버섯', 
          '토마토', '청상추', '백다다기', '애호박', '캠벨얼리', '샤인마스캇'
         ]

        df = pd.concat(data_list).reset_index(drop=True)
        target_df = df[df['KIND_NM'].isin(target) | df['PUM_NM'].isin(target)].reset_index(drop=True)
        target_df.loc[target_df['PUM_NM'].isin(target), "TY"] = target_df[target_df['PUM_NM'].isin(target)]['PUM_NM']
        target_df.loc[target_df['TY'].isnull() & target_df['KIND_NM'].isin(target), "TY"] = target_df[target_df['TY'].isnull() & target_df['KIND_NM'].isin(target)]['KIND_NM']
        target_set_df = target_df[target_df['TOT_QTY']>0].groupby(['SALEDATE', 'TY'])[['TOT_QTY','TOT_AMT']].sum()
        target_set_df = target_set_df.reset_index()
        target_set_df['PRICE'] = target_set_df['TOT_AMT']/target_set_df['TOT_QTY']
        price_df = target_set_df[['SALEDATE',"TY", "TOT_QTY", 'PRICE']]

        re_df = price_df.pivot_table(["TOT_QTY", "PRICE"], index="SALEDATE", columns=["TY"],aggfunc="max")
        re_df = re_df.reset_index()
        re_df.columns = ['date', "건고추_가격(원/kg)", "깻잎_가격(원/kg)", "당근_가격(원/kg)",
                        "대파_가격(원/kg)", "마늘_가격(원/kg)", "무_가격(원/kg)", "미나리_가격(원/kg)",
                        "배추_가격(원/kg)", "백다다기_가격(원/kg)", "새송이_가격(원/kg)", "샤인마스캇_가격(원/kg)",
                        "시금치_가격(원/kg)", "애호박_가격(원/kg)", "양배추_가격(원/kg)", "양파_가격(원/kg)", 
                        "얼갈이배추_가격(원/kg)", "청상추_가격(원/kg)", "캠벨얼리_가격(원/kg)", "토마토_가격(원/kg)",
                        "파프리카_가격(원/kg)", "팽이버섯_가격(원/kg)",
                        '건고추_거래량(kg)',"깻잎_거래량(kg)","당근_거래량(kg)","대파_거래량(kg)","마늘_거래량(kg)",
                        "무_거래량(kg)","미나리_거래량(kg)","배추_거래량(kg)","백다다기_거래량(kg)","새송이_거래량(kg)",
                        "샤인마스캇_거래량(kg)","시금치_거래량(kg)","애호박_거래량(kg)","양배추_거래량(kg)","양파_거래량(kg)",
                        "얼갈이배추_거래량(kg)","청상추_거래량(kg)","캠벨얼리_거래량(kg)","토마토_거래량(kg)","파프리카_거래량(kg)",
                        "팽이버섯_거래량(kg)"]
        re_df["요일"] = np.nan
        re_df = re_df[list(df_.columns)]
        re_df['date'] = re_df['date'].apply(lambda x:x[:4]+"-"+x[4:6]+"-"+x[6:8])

        data_set = pd.concat([df_, re_df], axis=0)
        data_set.reset_index(drop=True, inplace=True)
        print("데이터 수집 종료")

        return data_set
    else:
        return df_

def lgbm_inference(save_path):
    LEN_LAG = 28

    today = str(datetime.date.today().year) + "-" + str(datetime.date.today().month).zfill(2) + "-" + str(datetime.date.today().day).zfill(2)
    args = easydict.EasyDict({ 
                            'save_path' : save_path, 
                            })        
    data = add_data(args) 

    submission = pd.read_csv(args.save_path+'sample_submission.csv')
    test_df = fe(data, train=False)    

    unique_pum = [
        '배추', '무', '양파', '건고추','마늘',
        '대파', '얼갈이배추', '양배추', '깻잎',
        '시금치', '미나리', '당근',
        '파프리카', '새송이', '팽이버섯', '토마토',
    ]

    unique_kind = [
        '청상추', '백다다기', '애호박', '캠벨얼리', '샤인마스캇'
    ]   

    model_dict = {}
    split = 28 #validation

    kfold=KFold(n_splits=5)

    print('lgbm 모델 학습 시작')
    for pum in tqdm(unique_pum + unique_kind):
        # 품목 품종별 전처리
        temp_df = test_df[['date','요일',f'{pum}_거래량(kg)', f'{pum}_가격(원/kg)']]
        temp_df = preprocessing(temp_df, pum, len_lag=LEN_LAG)
    
        # 주차별(1,2,4w) 학습
        for week_num in [1,2,4] :
            x = temp_df[temp_df[f'{week_num}_week']>0].iloc[:,:-3]
            y = temp_df[temp_df[f'{week_num}_week']>0][f'{week_num}_week']
        
            fold_num = 0
            for train_index, test_index in kfold.split(x):
                #train, test split
                x_train = x.iloc[train_index]
                y_train = y.iloc[train_index]
                x_valid = x.iloc[test_index]
                y_valid = y.iloc[test_index] 
        
                model_dict[f'{pum}_model_{week_num}_{fold_num}'] = model_train(x_train, y_train, x_valid, y_valid)

                fold_num = fold_num + 1
    
    print('lgbm 모델 학습 완료')
    #추론
    private_date_list = submission[submission['예측대상일자'].str.contains('2021')]['예측대상일자'].str.split('+').str[0].unique()

    print('lgbm 모델 추론 시작')
    for pum in unique_pum + unique_kind:
        # 예측기준일에 대해 전처리
        temp_test = pd.DataFrame([{'date' : today}]) #예측기준일
        alldata = pd.concat([test_df,temp_test], sort=False).reset_index(drop=True)

        day = pd.to_datetime(today)
        alldata.iloc[-1, 1] = day.dayofweek
            
        alldata = alldata[['date','요일', f'{pum}_거래량(kg)', f'{pum}_가격(원/kg)']].fillna(0)
        alldata = alldata.iloc[-28:].reset_index(drop=True)
        alldata = preprocessing(alldata, pum, len_lag=LEN_LAG)
        temp_test = alldata.iloc[-1].astype(float)
         

        # 개별 모델을 활용하여 1,2,4주 후 가격 예측
        for week_num in [1,2,4] : 
            result_list = []  
            for fold_num in range(5): 
                temp_model = model_dict[f'{pum}_model_{week_num}_{fold_num}']
                result = temp_model.predict(temp_test)
                result_list.append(result[0]) 
                                
            condition = (submission['예측대상일자']==f'{today}+{week_num}week')
            idx = submission[condition].index 
            submission.loc[idx, f'{pum}_가격(원/kg)'] = np.mean(result_list) 

    submission.to_csv(args.save_path+f"output/lgbm_{today}.csv", index=False)

    return print("결과 파일이", args.save_path+f"output/lgbm_{today}.csv","에 저장됐습니다.")

