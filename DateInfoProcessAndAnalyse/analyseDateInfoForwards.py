import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('dateData.csv')

df2 = pd.read_csv('result.csv')

data = []
for i in range(len(df1)):
    date = df1.iloc[i, 0]
    if date in df2['date'].values:
        position = df2[df2['date'] == date].index[0]+2
    else:
        position = np.nan
    data.append({'date': date, 'position': position})

df3 = pd.DataFrame(data)
df3['loss_ratio'] = df3['date'].apply(lambda x: df2[df2['date'] == x]['loss_ratio'].values[0] if x in df2['date'].values and not df2[df2['date'] == x]['loss_ratio'].empty else np.nan)
df3.dropna(subset=['loss_ratio'], inplace=True)
df3['description'] = df3['date'].apply(lambda x:df1[df1['date']==x]['description'].values[0])
df3.sort_values('position', inplace=True)

df3.to_csv('dateDataPosition.csv', index=False)


df1=pd.read_csv('dateData.csv')
df2=pd.read_csv('result.csv')
df3=pd.read_csv('dateDataPosition.csv')

# 计算均值
mean_df3 = df3['loss_ratio'].mean()
mean_df2 = df2['loss_ratio'].mean()
print(f'mean_df3: {mean_df3}')
print(f'mean_df2: {mean_df2}')

# 绘制df3的loss_ratio的直方图,并且和df2的loss_ratio的直方图进行对比
plt.figure()
plt.hist(df3['loss_ratio'], bins=100, color='r', alpha=0.7, label=f'df3 (mean={mean_df3:.6f})')
plt.hist(df2['loss_ratio'], bins=100, color='b', alpha=0.7, label=f'df2 (mean={mean_df2:.6f})')
plt.legend()
plt.show()