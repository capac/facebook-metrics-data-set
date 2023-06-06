import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

plt.style.use('barplot-style.mplstyle')

fb_df = pd.read_csv('data/dataset_Facebook.csv', sep=',', na_values='NaN',
                    dtype={'Category': 'object', 'Paid': 'object',
                           'Post Month': 'object', 'Post Weekday': 'object',
                           'Post Hour': 'object'})

# plot data
type_data = fb_df.Type.value_counts(sort=False, dropna=False)
category_data = fb_df.Category.value_counts(sort=False, dropna=False).sort_index()
paid_data = fb_df.Paid.value_counts(sort=False, dropna=True).sort_index()
df_list = [type_data, category_data, paid_data]
labels_list = [type_data.index, category_data.index, ['Not paid', 'Paid']]
titles_list = ['Number of posts per type values',
               'Number of posts per category values',
               '''Number of posts for 'Not paid' / 'Paid' values''']

# plot (not used in project)
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), layout='constrained')
for ax, df, lb, tt in zip(axes.flat, df_list, labels_list, titles_list):
    ax.bar(x=df.index, height=df.values, color=colors)
    ax.set_title(tt)
    ax.set_xticks(ticks=df.index, labels=lb)
fig.savefig('plots/plots_ss/number_posts_per_category.png')


# first project plot
# data for first plot
month_data = fb_df['Post Month'].value_counts(sort=False)
sorted_index_m = [int(x) for x in list(month_data.index)]
month_data_new = pd.concat([pd.Series(sorted_index_m), pd.Series(month_data.values)], axis=1)
month_data_new.rename(columns={0: 'Index', 1: 'Frequency'}, inplace=True)
month_data_new.sort_values(by='Index', inplace=True)
# data for second plot
day_data = fb_df[['Post Weekday']].value_counts(sort=False).\
           sort_index().reset_index()
day_data.rename(columns={'Post Weekday': 'Index', 'count': 'Frequency'},
                inplace=True)
# data for third plot
hour_data = fb_df['Post Hour'].value_counts(sort=False)
sorted_index_h = [int(x) for x in list(hour_data.index)]
hour_data_new = pd.concat([pd.Series(sorted_index_h),
                           pd.Series(hour_data.values)], axis=1)
hour_data_new.rename(columns={0: 'Index', 1: 'Frequency'},
                     inplace=True)
hour_data_new.sort_values(by='Index', inplace=True)

df_list = [month_data_new, day_data, hour_data_new]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
          'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
labels_list = [months, days, hour_data_new.Index]
titles_list = ['Number of posts per month of the year',
               'Number of posts per day of the week',
               'Number of posts per hour of the day',
               ]

# final plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), layout='constrained')
for ax, df, lb, tt in zip(axes.flat, df_list, labels_list, titles_list):
    ax.bar(x=df.Index.values, height=df.Frequency.values, color=colors)
    ax.set_xticks(ticks=df.Index, labels=lb)
    ax.set_title(tt)
fig.savefig('plots/plots_ss/number_posts_per_time_interval_ss.png')


# second project plot
# code for first plot
tot_int_ph = fb_df['Total Interactions'].groupby(fb_df['Post Hour']).mean()
sorted_index = [int(x) for x in list(tot_int_ph.index)]
tot_int_ph_new = pd.concat([pd.Series(sorted_index), pd.Series(tot_int_ph.values)], axis=1)
tot_int_ph_new.rename(columns={0: 'Index', 1: 'Frequency'}, inplace=True)
tot_int_ph_new.sort_values(by='Index', inplace=True)

# code for second plot
tot_int_pw_new = fb_df['Total Interactions'].groupby(fb_df['Post Weekday']).\
                 mean().reset_index()
tot_int_pw_new.rename(columns={'Post Weekday': 'Index',
                               'Total Interactions': 'Frequency'},
                      inplace=True)
tot_int_pw_new.sort_values(by='Index', inplace=True)

# code for third plot
tot_int_pm = fb_df['Total Interactions'].groupby(fb_df['Post Month']).mean()
sorted_index = [str(x).zfill(2) for x in list(tot_int_pm.index)]
tot_int_pm_new = pd.concat([pd.Series(sorted_index),
                            pd.Series(tot_int_pm.values)], axis=1)
tot_int_pm_new.rename(columns={0: 'Index', 1: 'Frequency'}, inplace=True)
tot_int_pm_new.sort_values(by='Index', inplace=True)

# code for fourth plot
tot_int_p_new = fb_df['Total Interactions'].groupby(fb_df['Paid']).\
                mean().reset_index()
tot_int_p_new.rename(columns={'Paid': 'Index',
                              'Total Interactions': 'Frequency'},
                     inplace=True)
tot_int_p_new.sort_values(by='Index', inplace=True)

# dataframes
df_list = [tot_int_ph_new, tot_int_pw_new, tot_int_pm_new, tot_int_p_new]
# labels
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
          'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
paid_or_not = ['Not paid', 'Paid']
labels_list = [tot_int_ph_new.Index, days, months, paid_or_not]
# titles
titles_list = ['Average number of total interactions per hour of the day',
               'Average number of total interactions per day of the week',
               'Average number of total interactions per month of the year',
               'Average number of total interactions per paid advertisement',
               ]

# final plot
fig, axes = plt.subplots(nrows=2, ncols=2)
for ax, df, lb, tt in zip(axes.flat, df_list, labels_list, titles_list):
    ax.bar(x=df.Index, height=df.Frequency, color=colors,
           yerr=np.std(df.Frequency.values))
    ax.set_title(tt)
    ax.set_xticks(ticks=df.Index, labels=lb)
    # ax.yaxis.set_tick_params(labelsize=9)
fig.savefig('plots/plots_ss/average_number_of_total_interactions_ss.png')


# third project plot
numerical_cols = ['Page total likes', 'Lifetime Post Total Reach',
                  'Lifetime Post Total Impressions', 'Lifetime Engaged Users',
                  'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                  'Lifetime Post Impressions by people who have liked your Page',
                  'Lifetime Post reach by people who like your Page',
                  'Lifetime People who have liked your Page and '
                  'engaged with your post',
                  'comment', 'like', 'share', 'Total Interactions',]

scalar = StandardScaler()
fb_perf_df = fb_df[numerical_cols]
fb_std = scalar.fit_transform(fb_perf_df)
fb_std_df = pd.DataFrame(fb_std, columns=numerical_cols)

# final plot
fig, axes = plt.subplots(nrows=6, ncols=2, layout='constrained')
for ax, clm, col in zip(axes.flat, numerical_cols[1:], colors):
    ax.hist(np.log(fb_std_df.loc[:, clm]+1), color=col, bins=60,)
    ax.set_title(clm.capitalize())
plt.suptitle("Logarithmic standardized performance metrics", fontweight='bold')
plt.savefig('plots/plots_ss/performance_metrics_log_hist_ss.png')
