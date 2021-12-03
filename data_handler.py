import csv
import datetime
from pandas import read_csv

df = read_csv('raw_data.csv')

day = df.iloc[0]['generation_time']
flow = 0
speed = 0
count = 0

with open('dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['flow', 'speed', 'weekday'])

    for index, row in df.iterrows():
        if row['generation_time'] == day:
            flow += row['flow']
            speed += row['speed']
            count += 1
        else:
            [str, zone] = day.split('+')
            str = str.replace('T', ' ')
            timestamp = datetime.datetime.strptime(str, "%Y-%m-%d %H:%M:%S")
            writer.writerow([round(flow / count, 2), round(speed / count, 2),
                           timestamp.weekday()])
            day = row['generation_time']
            flow = row['flow']
            speed = row['speed']
            count = 1
