import ssl
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv('https://archive.ics.uci.edu/static/public/9/data.csv')
#df.info()

# 删除指定的列
df.drop(columns=['car_name'], inplace=True)
# 计算相关系数矩阵
print(df.corr())

# 删除有缺失值的样本
df.dropna(inplace=True)
# 将origin字段处理为类别类型
df['origin'] = df['origin'].astype('category') 
# 将origin字段处理为独热编码
df = pd.get_dummies(df, columns=['origin'], drop_first=True)
df