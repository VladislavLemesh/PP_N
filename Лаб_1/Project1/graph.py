import matplotlib.pyplot as plt
import pandas

plt.figure(figsize=(8,6))

data = pandas.read_csv("output.csv")
length = data.shape[0]

plt.plot(data["T"], [float(i) for i in data["Duration"]], label="Время выполнения", c="blue")

plt.xlabel('Количество потоков')
plt.ylabel('Время выполнения, с')
plt.legend()
plt.show()