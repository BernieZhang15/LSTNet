from pandas import read_csv
from matplotlib import pyplot
from fbprophet import Prophet

df = read_csv('raw_data.csv', header=0, index_col=0).values
pyplot.plot(df)
pyplot.show()

# df_train = df.iloc[:50000, 0: 2]
# df_test = df.iloc[50000: , 0: 1]
#
# df_train.columns = ['ds', 'y']
# df_test.columns = ['ds']
#
# model = Prophet()
# model.fit(df_train)
#
# forecast = model.predict(df_test)
# model.plot(forecast)
# pyplot.show()









