import matplotlib.pyplot as plt
import pandas

plt.figure(figsize=(8,6))

data = pandas.read_csv("output.csv")
length = data.shape[0]

t = data[0:0 + length]["T"]
duration = data[0:0 + length]["Duration"]
duration = [float(i) for i in duration]
plt.plot(t, duration, c="blue")

plt.title("Зависимость времени выполнения FFT от числа потоков")
plt.xlabel('Количество потоков')
plt.ylabel('Время выполнения, мс')
plt.xticks(t)

for (xi, yi) in zip(t, duration):
    plt.text(xi, yi, yi, va='bottom', ha='left')
plt.vlines(t, 0, duration, linestyle="dashed")
plt.hlines(duration, 0, t, linestyle="dashed")

plt.show()