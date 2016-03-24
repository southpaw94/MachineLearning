import pymysql
import pandas as pd

conn = pymysql.connect(host='localhost', 
        user='monitor', 
        password='password', 
        db='ECE491')
df = pd.read_sql("SELECT * FROM sensorData", conn)
print(df.columns)
