import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 30})


width = 0.12
yLim = 90
# yLabel = "number of CNOT gates"
# title = "Compare number of CNOT gates"

yLabel = "Depth"
title = "Compare depth between configurations"


# names = ["Adder 10 qubits", "AQFT 6 qubits"]
# if "CNOT" in title:
#     i1 = [69,0]
#     ti1 = [117,45]
#     ti2 = [120,45]
#     ti3 = [117,45]
#     tc1 = [59,21]
#     tc2 = [58,18]
#     tc3 = [59,18]
# else: #depth
#     i1 = [97,12]
#     ti1 = [186,64]
#     ti2 = [180,65]
#     ti3 = [186,64]
#     tc1 = [94,32]
#     tc2 = [97,32]
#     tc3 = [94,32]


# names = ["GHZ 24 qubits", "HWE 10 qubits"]
# if "CNOT" in title:
#     i1 = [23,9]
#     ti1 = [68,9]
#     ti2 = [62,9]
#     ti3 = [68,9]
#     tc1 = [11,4]
#     tc2 = [11,4]
#     tc3 = [11,4]
# else: #depth
#     i1 = [25,14]
#     ti1 = [66,13]
#     ti2 = [60,13]
#     ti3 = [66,13]
#     tc1 = [15,10]
#     tc2 = [15,10]
#     tc3 = [15,10]


# names = ["Supremacy 12 qubits", "Sycamore 12 qubits"]
# if "CNOT" in title:
#     i1 = [0,0]
#     ti1 = [59,8]
#     ti2 = [56,8]
#     ti3 = [59,8]
#     tc1 = [16,4]
#     tc2 = [16,4]
#     tc3 = [16,4]
# else: #depth
#     i1 = [11,5]
#     ti1 = [63,12]
#     ti2 = [52,12]
#     ti3 = [54,12]
#     tc1 = [28,12]
#     tc2 = [32,12]
#     tc3 = [32,12]


names = ["Supremacy 20 qubits", "Supremacy 25 qubits"]
if "CNOT" in title:
    i1 = [0,0]
    ti1 = [103,139]
    ti2 = [109,151]
    ti3 = [118,148]
    tc1 = [40,57]
    tc2 = [40,57]
    tc3 = [37,60]
else: #depth
    i1 = [11,11]
    ti1 = [47,69]
    ti2 = [69,75]
    ti3 = [58,68]
    tc1 = [50,55]
    tc2 = [39,48]
    tc3 = [39,57]

######################################################################################
label1 = "input circuit"
label2 = "transpiled input circuit"
label3 = "maximum from transpiled subcircuits"

x = np.arange(len(names))

fig, ax = plt.subplots()

rectsi1 = ax.bar(x-width*3, i1, width, color='cyan', label=label1, edgecolor="black")
rectsti1 = ax.bar(x-width*2, ti1, width, color='red', label=label2, edgecolor="black")
rectsti2 = ax.bar(x-width*1, ti2, width, color='red', edgecolor="black")
rectsti3 = ax.bar(x-width*0, ti3, width, color='red', edgecolor="black")
rectstc1 = ax.bar(x+width*1, tc1, width, color='green', label=label3, edgecolor="black")
rectstc2 = ax.bar(x+width*2, tc2, width, color='green', edgecolor="black")
rectstc3 = ax.bar(x+width*3, tc3, width, color='green', edgecolor="black")

ax.bar_label(rectsi1, padding=3)
ax.bar_label(rectsti1, padding=3)
ax.bar_label(rectsti2, padding=3)
ax.bar_label(rectsti3, padding=3)
ax.bar_label(rectstc1, padding=3)
ax.bar_label(rectstc2, padding=3)
ax.bar_label(rectstc3, padding=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x, names) 
ax.set_ylabel(yLabel)
ax.set_title(title)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, yLim)

plt.show()