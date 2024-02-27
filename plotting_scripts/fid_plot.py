import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 34})

# names = ["Adder 10 qubits", "AQFT 6 qubits", "GHZ 24 qubits"]
# i = [0.74,0.99, 0.99]
# c = [0.77,0.95, 0.8]

# names = ["HWE 10 qubits", "Supremacy 12 qubits", "Sycamore 12 qubits"]
# i = [0.87, 0.94, 0.14]
# c = [0.92, 0.76, 0.93]

names = ["Supremacy 20 qubits", "Supremacy 25 qubits"]
i = [1.27E-05, 1.97E-31]
c = [0.002, 1]

label1 = "original circuit"
label2 = "cut circuit"

x = np.arange(len(names))
width = 0.4

fig, ax = plt.subplots()

rectsi1 = ax.bar(x-width/2, i, width, color='cyan', label=label1, edgecolor="black")
rectsti1 = ax.bar(x+width/2, c, width, color='green', label=label2, edgecolor="black")

ax.bar_label(rectsi1, padding=3)
ax.bar_label(rectsti1, padding=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x, names)
ax.set_ylabel('Fidelity')
ax.set_title(f"Compare fidelity between circuits")
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1.1)

plt.show()