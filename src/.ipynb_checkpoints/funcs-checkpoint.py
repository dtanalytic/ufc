

def get_rolling_stats(df_fighter, op_d, min_fights_num):
#     import pdb; pdb.set_trace()
    df_stat = df_fighter.expanding(min_periods=min_fights_num).agg(op_d)
    df_stat['nlosses_stat_rol'] = df_fighter['lose_stat'].rolling(window=min_fights_num).sum()
    
    return df_stat.shift(1)