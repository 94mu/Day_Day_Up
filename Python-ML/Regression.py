import ssl
import pandas as pd
import sklearn

ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv('https://archive.ics.uci.edu/static/public/9/data.csv')
#df.info()

# 删除指定的列
df.drop(columns=['car_name'], inplace=True)
# 计算相关系数矩阵
#print(df.corr())

# 删除有缺失值的样本
df.dropna(inplace=True)
# 将origin字段处理为类别类型
df['origin'] = df['origin'].astype('category') 
# 将origin字段处理为独热编码
df = pd.get_dummies(df, columns=['origin'], drop_first=True)

# 将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split

X, y = df.drop(columns='mpg').values, df['mpg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=3)

# 模型训练
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
#print('回归系数:', model.coef_)
#print('截距:', model.intercept_)

y_pred = model.predict(X_test)

# 模型评估
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'均方误差: {mse:.4f}')
print(f'平均绝对误差: {mae:.4f}')
print(f'决定系数: {r2:.4f}')