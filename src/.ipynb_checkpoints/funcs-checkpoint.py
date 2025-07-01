import pandas as pd

def get_stat_feat(df, aggs, cust_aggs, cols, min_per_num, rol_window_size):

    rol_df = df.rolling(min_periods=min_per_num, window=rol_window_size).agg({it:aggs for it in cols})
    rol_df.columns = rol_df.columns.map(lambda x: f'{x[0]}_rol_{x[1]}')


    rol_cust_df_l = []
    for fn, func in cust_aggs:
        rol_cust_df_l.append(df.rolling(min_periods=min_per_num, window=rol_window_size).agg({colnm:
                                                                        lambda x: func(x) for colnm in cols})\
                              .rename(columns={colnm:f'{colnm}_rol_{fn}' for colnm in cols}))
    rol_cust_df = pd.concat(rol_cust_df_l, axis=1)


    exp_df = df.expanding(min_periods=min_per_num).agg({it:aggs for it in cols})
    exp_df.columns = exp_df.columns.map(lambda x: f'{x[0]}_exp_{x[1]}')

    exp_cust_df_l = []
    for fn, func in cust_aggs:
        exp_cust_df_l.append(df.expanding(min_periods=min_per_num).agg({colnm: lambda x: func(x) for colnm in cols})\
              .rename(columns={colnm:f'{colnm}_exp_{fn}' for colnm in cols}))

    exp_cust_df = pd.concat(exp_cust_df_l, axis=1)

    return pd.concat([rol_df, exp_df, rol_cust_df, exp_cust_df], axis=1).shift(1)

