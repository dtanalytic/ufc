import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import OneHotEncoder
import joblib

import sys
sys.path.append('.')
from src.prep_funcs import prep_strikes_fn, prep_fighters_fn
from src.stat_funcs import get_stat_feat, last_el

def get_features_event(ev_nm, fights_fn, fighters_fn, coef_fn, vocab_fn, ufc_rankings_fn, rankings_big_fn, conf, pred_df=pd.DataFrame()):
    '''
    Args:
        - ev_nm - название турнира для формирования признаков
        - fights_fn - пусть к файлу с боями
        - fighters_fn - пусть к файлу с бойцами
        - coef_fn - путь к обработанному файлу со ставками в event-ах
        - vocab_fn - путь к словарю после предобработки downloads с притянутыми версиями из датасетов и верификацией name_rus	fighters_name	fights_name	ranks_name	manual_verified	auto_verified
        - ufc_rankings_fn - путь к файлу с рейтингами ufc
        - rankings_big_fn - путь к файлу с неоф рейтингами ufc, но более детальными
        - conf - словарь по params.yaml
    Returns: 
        возвращает feature matrix для прогноза event-а
    '''
    coef_df = pd.read_excel(coef_fn)
    coef_df = coef_df.assign(event_day=lambda x:pd.to_datetime(x['date'], format='%d.%m.%Y'))
    if not ev_nm:
        if pred_df.shape[0]==0:
            pred_df = coef_df[coef_df['event_day']==coef_df['event_day'].max()]
    else:
        pred_df = coef_df[coef_df['name']==ev_nm]

    vocab_df = pd.read_csv(vocab_fn)

#     # создаем копии для "двойного" предикта
#     pred_df = pd.concat([pred_df,
# pred_df.rename(columns={'fighter1':'fi2', 'fighter2':'fi1', 'coef1':'co2', 'coef2':'co1'})\
#         .rename(columns={'fi2':'fighter2', 'fi1':'fighter1', 'co2':'coef2', 'co1':'coef1'})], ignore_index=True)

    
    pred_df = pred_df.merge(vocab_df[['name_rus','fighters_name', 'fights_name', 'ranks_name']].rename(columns={**{'name_rus':'fighter1'}, **{it:f'{it}1' for it in ['fighters_name', 'fights_name', 'ranks_name']}})
                            , on='fighter1', how='left')\
            .merge(vocab_df[['name_rus','fighters_name', 'fights_name', 'ranks_name']].rename(columns={**{'name_rus':'fighter2'}, **{it:f'{it}2' for it in ['fighters_name', 'fights_name', 'ranks_name']}}), on='fighter2', how='left')
    
    pred_df = pred_df.assign(event_day=pd.to_datetime(pred_df['event_day'], format='%Y-%m-%d').dt.to_period(freq='D'))


    # ----------
    # подгружаем ранги с датой <= событию и максимальной из оставшихся
    event_date = pred_df['event_day'].iloc[0]
    
    rank_df = pd.read_csv(ufc_rankings_fn).loc[lambda x: ~x['division_name'].isin(['Вне зависимости от категорий Top Rank', 'Женский, вне весовых категорий Top Rank'])]
    rank_df['date'] = pd.to_datetime(rank_df['date'], format='%Y-%m-%d').dt.to_period(freq='D')
    rank_df = rank_df.groupby(['name', 'date'], as_index=False)['rank'].min()
    
    rank_df = rank_df[rank_df['date']<event_date - 1]
    rank_df = rank_df[rank_df['date'] == rank_df['date'].max()]

    assert rank_df['name'].duplicated().sum()==0

    pred_df = pred_df.merge(rank_df[['name', 'rank']].rename(columns={'name':'fighter1', 'rank':'rank1'}), on='fighter1', how='left')\
       .merge(rank_df[['name', 'rank']].rename(columns={'name':'fighter2', 'rank':'rank2'}), on='fighter2', how='left')

    
    rank_big_df = pd.read_csv(rankings_big_fn, dtype='str')
    rank_big_df['date'] = pd.to_datetime(rank_big_df['date'], format='%Y-%m-%d').dt.to_period(freq='D')
    rank_big_df['name'] = rank_big_df['name'].str.replace(r'"\s*[^"]*"\s*', '', regex=True)
    rank_big_df['rank'] = rank_big_df['rank'].astype('float')
    
    rank_big_df = rank_big_df[rank_big_df['date']<event_date-1]
    rank_big_df = rank_big_df[rank_big_df['date']==rank_big_df['date'].max()]
    
    # считаем итоговый рейтинг по лучшему значению
    rank_big_df = rank_big_df.groupby(['name', 'date'], as_index=False)['rank'].min()
    assert rank_big_df['name'].duplicated().sum()==0

    pred_df = pred_df.merge(rank_big_df[['name', 'rank']].rename(columns={'name':'ranks_name1', 'rank':'rank_big1'}), on='ranks_name1', how='left')\
       .merge(rank_big_df[['name', 'rank']].rename(columns={'name':'ranks_name2', 'rank':'rank_big2'}), on='ranks_name2', how='left')

    # -----------
    # признак левого угла
    pred_df['left_corner'] = np.nan
    sel_ranks = pred_df['rank1'].notna()&pred_df['rank2'].notna()
    sel_big_ranks = pred_df['rank_big1'].notna()&pred_df['rank_big2'].notna()
    pred_df['left_corner_little'] = np.where((pred_df['rank1']<pred_df['rank2'])&sel_ranks, 1, np.where(sel_ranks, 0, np.nan))
    pred_df['left_corner_big'] = np.where((pred_df['rank_big1']<pred_df['rank_big2'])&(sel_big_ranks), 1, np.where(sel_big_ranks, 0, np.nan))
    pred_df['left_corner'] = pred_df['left_corner_little'].mask(pred_df['left_corner_little'].isna(), pred_df['left_corner_big'])

    # ------------------
    # поединки загружаем
    fighters_in = pd.concat([pred_df[['fights_name1']].rename(columns={'fights_name1':'name'}), 
                         pred_df[['fights_name2']].rename(columns={'fights_name2':'name'})]).dropna()['name'].drop_duplicates().tolist()
    
    fights_df = prep_strikes_fn(fights_in_fn=fights_fn, fighters=fighters_in)
    fights_df = fights_df[fights_df['event_date']<event_date-1]

    fights_df = fights_df.sort_values(by=['Fighter', 'event_date'])
    stat_cols = [it for it in fights_df.columns if re.search('_stat', it)]
    aggs = eval(conf['stat_feat_gen']['aggs'])
    cust_aggs = eval(conf['stat_feat_gen']['cust_aggs'])

    # добавляем так как это ожидаемый бои относительно каждого бойца, чтобы rolling характеристики посчитались для всех фактических боев (учитывая сдвиг)
    fights_add_df = pd.concat([pred_df[['fights_name1', 'fights_name2', 'event_day', 'name']].rename(columns={'fights_name1':'Fighter', 'fights_name2':'Opponent', 'event_day':'event_date', 'name':'Event'}),
           pred_df[['fights_name2', 'fights_name1', 'event_day', 'name']].rename(columns={'fights_name2':'Fighter', 'fights_name1':'Opponent', 'event_day':
                                                                                 'event_date', 'name':'Event'})], ignore_index=True)
    fights_df = pd.concat([fights_df, fights_add_df], ignore_index=True)

    # подсчет статистик для боев
    fights_stat_df = fights_df.groupby('Fighter').apply(lambda x: get_stat_feat(x, aggs = aggs, cust_aggs=cust_aggs, cols = stat_cols,
                                                    min_per_num=conf['stat_feat_gen']['min_fights_num'],
                                                    rol_window_size=conf['stat_feat_gen']['rol_window_size']).iloc[[-1]])

    # подсчет статистик с нулевым сдвигом
    fights0_df = fights_df[['Fighter', 'event_date']].copy()
    
    fights0_df['days_nofight_stat'] = fights0_df.groupby('Fighter', as_index=False)['event_date'].diff().map(lambda x: x.n, na_action='ignore')
    fights0_df['days_nofight_stat'] = fights0_df['days_nofight_stat'].replace({pd.NaT: np.nan}).astype('float32')

    fights_stat0_df = fights0_df.groupby('Fighter').apply(lambda x: get_stat_feat(x, aggs = aggs, cust_aggs=cust_aggs, cols = ['days_nofight_stat'], min_per_num=conf['stat_feat_gen']['min_fights_num'], rol_window_size=conf['stat_feat_gen']['rol_window_size'], n_shift=0).iloc[[-1]])    
    
    fights_stat0_df = fights_stat0_df.assign(index_col = fights_stat0_df.index.map(lambda x: x[1])).merge(fights0_df[['days_nofight_stat']], left_on='index_col', right_index=True, how='left')\
                    .drop(columns='index_col')
    
    
    fights_stat_df.index = fights_stat_df.index.map(lambda x: x[0])
    fights_stat0_df.index = fights_stat0_df.index.map(lambda x: x[0])
    
    fights_stat = fights_stat0_df.merge(fights_stat_df, left_index=True, right_index=True).reset_index(names='full_name')

    # ---------------
    # добавляем признаки бойцов

    fighters_in = pd.concat([pred_df[['fighters_name1']].rename(columns={'fighters_name1':'name'}), pred_df[['fighters_name2']].rename(columns={'fighters_name2':'name'})]).dropna()['name'].drop_duplicates().tolist()

    
    # fighters_df = prep_fighters_fn(fighters_fn, fighters=fighters_in)
    fighters_df = prep_fighters_fn(fighters_fn, fighters=fighters_in).drop_duplicates(subset='full_name')

    fighters_df['dob_diff_stat_custom'] = fighters_df['dob'].map(lambda x: x.start_time.timestamp()/(30*24*3600) if not pd.isnull(x) else np.nan)
    
    stat_df = fighters_df.merge(fights_stat, on='full_name', how='left')
    
    # переименовать колонки
    stat_df = stat_df.rename(columns={'height_stat':'height_stat_custom', 'reach_stat':'reach_stat_custom'})

    
    feat_cols = [it for it in stat_df.columns if '_stat_rol' in it or '_stat_exp' in it or '_stat_custom' in it]

    # ---------------
    # формируем признаки боев
    ind_cols = ['name', 'fighter', 'opponent', 'event_day', 'coef1', 'coef2']
    
    df11 = pred_df.merge(stat_df, how='left', left_on='fighters_name1', right_on='full_name')\
        .rename(columns={'fighters_name1':'fighter', 'fighters_name2':'opponent'})[ind_cols+feat_cols]\
        .set_index(ind_cols)

    df12 = pred_df.merge(stat_df, how='left', left_on='fighters_name2', right_on='full_name')\
            .rename(columns={'fighters_name1':'fighter', 'fighters_name2':'opponent'})[ind_cols+feat_cols]\
            .set_index(ind_cols)
    
    df21 = pred_df.merge(stat_df, how='left', left_on='fighters_name2', right_on='full_name')\
            .rename(columns={'fighters_name2':'fighter', 'fighters_name1':'opponent', 'coef1':'c1', 'coef2':'c2'})\
            .rename(columns={'c1':'coef2', 'c2':'coef1'})\
            [ind_cols+feat_cols]\
            .set_index(ind_cols)
    
    df22 = pred_df.merge(stat_df, how='left', left_on='fighters_name1', right_on='full_name')\
            .rename(columns={'fighters_name2':'fighter', 'fighters_name1':'opponent', 'coef1':'c1', 'coef2':'c2'})\
            .rename(columns={'c1':'coef2', 'c2':'coef1'})\
            [ind_cols+feat_cols]\
            .set_index(ind_cols)
    
    
    
    pred_fightes_df = pd.concat([(df11 - df12).dropna(),(df21 - df22).dropna()] )

    
    pred_fightes_df = pred_fightes_df.reset_index().merge(fights_stat0_df.reset_index(names='fighter')[['fighter', 'days_nofight_stat']], on='fighter', how='left')\
                    .rename(columns={'days_nofight_stat':'fighter_days_nofight_custom'})\
                    .merge(fights_stat0_df.reset_index(names='fighter')[['fighter', 'days_nofight_stat']].rename(columns={'fighter':'opponent'}), on='opponent', how='left')\
                    .rename(columns={'days_nofight_stat':'opponent_days_nofight_custom'})\
                    .set_index(ind_cols)
    
    pred_fightes_df.columns=[f"{it.replace('_stat', '')}_feat" for it in pred_fightes_df.columns]

    pred_fightes_df = stat_df.rename(columns={'full_name':'fighter'})[['fighter', 'stance',	'dob']].merge(pred_fightes_df.reset_index(), on='fighter', how='right')
    pred_fightes_df = pred_fightes_df.assign(fighter_dob_custom_feat = lambda x: (x['event_day'] - x['dob']).map(lambda y: y.n/30))\
                    .rename(columns={'stance':'fighter_stance_custom_feat'})\
                    .drop(columns=['dob'])
    
    # pred_fightes_df = stat_df.rename(columns={'full_name':'opponent'})[['opponent', 'stance']].merge(pred_fightes_df, on='opponent', how='right')\
    #         .rename(columns={'stance':'opponent_stance_custom_feat'})

    pred_fightes_df = stat_df.rename(columns={'full_name':'opponent'})[['opponent', 'stance', 'height_stat_custom', 'reach_stat_custom']].merge(pred_fightes_df, on='opponent', how='right')\
        .rename(columns={'stance':'opponent_stance_custom_feat', 'height_stat_custom':'opponent_height_custom_feat', 
                         'reach_stat_custom':'opponent_reach_custom_feat', 'height_custom_feat':'height_diff_custom_feat', 'reach_custom_feat':'reach_diff_custom_feat'})
    

    pred_fightes_df = pred_fightes_df.merge(pred_df[['fighters_name1', 'left_corner']].rename(columns={'left_corner':'left_corner_custom_feat', 'fighters_name1':'fighter'}), how='left', on='fighter')
    
    feats = [it for it in pred_fightes_df.columns if '_feat' in it]


    
    pred_fightes_df = pred_fightes_df[ind_cols+feats]
    pred_fightes_df.columns = [it.lower() for it in pred_fightes_df.columns]
    
    fighter_feats = ['left_corner_custom_feat', 'height_diff_custom_feat', 'reach_diff_custom_feat', 'dob_diff_custom_feat', 
         'fighter_stance_custom_feat', 'opponent_stance_custom_feat', 'fighter_dob_custom_feat', 'fighter_days_nofight_custom_feat', 'opponent_days_nofight_custom_feat',
                'opponent_height_custom_feat', 'opponent_reach_custom_feat']
    
    other_feats = sorted([it for it in pred_fightes_df.columns if not it in fighter_feats and not it in ind_cols])
    
    pred_fightes_df = pred_fightes_df[ind_cols + fighter_feats + other_feats]


    ohe = joblib.load(conf['split_data']['ohe_fn'])

    pred_fightes_df = pred_fightes_df.drop(columns=['fighter_stance_custom_feat', 'opponent_stance_custom_feat'])\
        .merge(pd.DataFrame(ohe.transform(pred_fightes_df[['fighter_stance_custom_feat', 'opponent_stance_custom_feat']]), 
                     index=pred_fightes_df.index, columns=ohe.get_feature_names_out()),
        left_index=True, right_index=True).rename(columns={'name':'event'})


    # признак левого угла добавляем для "двойников"
    pred_fightes_df = pred_fightes_df.assign(fighters = pred_fightes_df.apply(lambda x: ' '.join(sorted([x['fighter'], x['opponent']])) , axis=1))
    pred_fightes_df = pred_fightes_df.merge(pred_fightes_df[['fighters', 'fighter','left_corner_custom_feat']].rename(columns={'left_corner_custom_feat':'lc', 'fighter':'fi'}), on='fighters', how='left')\
                    .loc[lambda x: x['fighter']!=x['fi']]
    pred_fightes_df['left_corner_custom_feat'] = pred_fightes_df['left_corner_custom_feat'].mask(
    pred_fightes_df['left_corner_custom_feat'].isna(), 1-pred_fightes_df['lc'])
    pred_fightes_df = pred_fightes_df.drop(columns=['fighters', 'fi', 'lc'])

    
    return pred_fightes_df
    
def place_bet(placebet_df, strategy_selection, alpha):
        
    df = placebet_df.copy()
    # переводим коэффициенты в вероятности с учетом излишка на маржу букмекера
    df['proba1'] = 1/df['coef1']
    df['proba2'] = 1/df['coef2']
    df['proba_sum_with_marja'] = df[['proba1', 'proba2']].sum(axis=1)
    df['proba1'] = df['proba1']/df['proba_sum_with_marja'] 
    df['proba2'] = df['proba2']/df['proba_sum_with_marja']
    
    # ищем недооцененных и profit складываем в колонку diff 
    df['score2'] = 1 - df['score1']
    df['diff1'] = df['proba1'] - df['score1']
    df['diff2'] = df['proba2'] - df['score2']
    
    df['selector'] = np.where(df['diff1']>0, 2, 1)    
    df['diff'] = np.where(df['selector']==1, df['diff1'], df['diff2'])
    
    # среди недооцененных букмекером ищем те, где алгоритм предсказал победу недооцененного бойца
    # это лучшая ставка
    # alpha*x - на бои с proba > 0.5, (1-alpha)*x на бои с proba<0.5; x=1, alpha=0.7
    df['betwin'] = df['score1'].mask(df['selector']==2, df['score2'])>0.5

    alpha_sel = df['betwin'] & strategy_selection
    # приграничные случаи для alpha
    if alpha_sel.sum()==0:
        alpha = 0
    elif alpha_sel.sum()==df.shape[0]:
        alpha = 1
    
    # среди betwin-ов распределяем alpha денег
    df.loc[alpha_sel, 'bet'] = (df.loc[alpha_sel, 'diff'].abs()/df.loc[alpha_sel, 'diff'].abs().sum()).map(lambda x: x*alpha)
    # на остальных недооцененных распределяем 1-alpha денег
    df.loc[(~df['betwin']) & strategy_selection, 'bet'] = (df.loc[(~df['betwin']) & strategy_selection, 'diff'].abs()/df.loc[(~df['betwin'])&strategy_selection, 'diff'].abs().sum()).map(lambda x: x*(1-alpha))

    return df
 
def calc_profit(placebet_df, strategy_selection, alpha):
    '''
    Args:
    df - dataframe with:
        target - column name with target
        score - calibrated_proba to compare with bookmakers proba
    strategy_selection - series with condition that fits playing strategy
    
    Returns:
    dataframe that fits input dataframe index with values of profit for every row

    Example:
    '''
    bet_df = place_bet(placebet_df, strategy_selection = strategy_selection, alpha=alpha)
    # target==1 и selector==1, выиграл левый и на него ставили, target==0 и selector==2, выиграл правый и на него ставили, 
    bet_df['income'] = np.where((bet_df['target']+bet_df['selector']).map(lambda x: x==2), 
                               bet_df['bet']*bet_df['coef1'].mask(bet_df['selector']==2, bet_df['coef2']), 0)
    
    return bet_df['income'].sum(), bet_df


def calc_time_profit(placebet_df, strategy_selection, alpha):
    '''
    Args:
        event_day - колонка в placebet_df со значениями pd.Period заданной частоты (будет группировка по каждому уникальному периоду)
    Returns:
        income by time
    '''
    res_l = []
    for event_day in placebet_df.event_day.unique():
        sel = (placebet_df.event_day==event_day)
        t_df = placebet_df.loc[sel]#.reset_index(drop=True)
        income, df = calc_profit(placebet_df=t_df, strategy_selection=strategy_selection, alpha=alpha)
        res_l.append((event_day, t_df.shape[0], income, df))
    
    income_time_df = pd.DataFrame([(it[0], it[1], it[2]) for it in res_l], columns=['event_day', 'bet_num', 'income'])
    return income_time_df, res_l
    
