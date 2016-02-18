import pymysql
import pandas as pd

conn = pymysql.connect(host='localhost', 
        user='root', 
        password='password', 
        db='seniordesign')
df = pd.read_sql("SELECT * FROM sensorData", conn)
print(df.columns)
