import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 34})
yLim = 0.003
width = 0.4

# names = ["Adder 10 qubits", "AQFT 6 qubits", "GHZ 24 qubits"]
# i = [0.87, 0.96, 0.73]
# c = [0.99, 0.95, 0.99]

# names = ["HWE 10 qubits", "Supremacy 12 qubits", "Sycamore 12 qubits"]
# i = [0.86, 0.12, 0.18]
# c = [0.98, 0.78, 0.96]

# names = ["Supremacy 20 qubits", "Supremacy 25 qubits"]
# i = [1.27E-05, 1.97E-31]
# c = [0.002, 1]

names = ["Supremacy 20 qubits"]
i = [1.27E-05]
c = [0.002]

label1 = "original circuit"
label2 = "cut circuit"

x = np.arange(len(names))

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
ax.set_ylim(0, yLim)

plt.show()