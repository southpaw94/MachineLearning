import pandas as pd
import numpy as np

df = pd.read_csv('data.csv', header=None)
df.columns = ['mag', 'time']

array = df.iloc[:, :].values

peaks = []
times = []

for i in range(1, array.shape[0] - 1):
    if array[i - 1, 0] <= array[i, 0] and array[i, 0] >= array[i + 1, 0]:
        if array[i - 1, 0] != array[i, 0]:
            times.append(array[i, 1])
            peaks.append(array[i, 0])

freqs = []
for i in range(1, len(times)):
    freqs.append(1 / (times[i] - times[i - 1]))

freqs = np.array(freqs)
freqs *= 10**6

print(times)
print(peaks)
print(freqs)
