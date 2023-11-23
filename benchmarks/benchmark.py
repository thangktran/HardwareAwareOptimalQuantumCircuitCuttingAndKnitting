import pathlib
import sys
import math
from random import randrange
import datetime
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import EfficientSU2
from qiskit.providers.fake_provider import FakeKolkataV2

from helper_functions import generate_circ

from HwAwareCutter.Cutter import Cutter
from HwAwareCutter import Utilities
from HwAwareCutter.Logger import Logger


BENCHMARK_MAX_PARTITIONS = 2
BENCHMARK_MAX_N_QUBITS = 10
BENCHMARK_DIR = ""
CIRC_NAME = "ghz"
CIRC_N_QUBITS = 5
CIRC_DEPTH = 1

# usage: python benchmark.py -p 2 -q 10 [ran|sup|su|ghz|syc|hwe|bv|qft|aqft|add|erd] <nQubit> <nDepth>
# su doesn't need nDepth parameter, but a dummy 0 should be use
if len(sys.argv) == 8 and sys.argv[1] == "-p" and sys.argv[3] == "-q":
    BENCHMARK_MAX_PARTITIONS = int(sys.argv[2])
    BENCHMARK_MAX_N_QUBITS = int(sys.argv[4])
    CIRC_NAME = str(sys.argv[5]).lower()
    CIRC_N_QUBITS = int(sys.argv[6])
    CIRC_DEPTH = int(sys.argv[7])

BENCHMARK_DIR = f"./benchmark_results/{CIRC_NAME}_{CIRC_N_QUBITS}_{CIRC_DEPTH}_{BENCHMARK_MAX_PARTITIONS}_{BENCHMARK_MAX_N_QUBITS}_{datetime.datetime.now()}"
pathlib.Path(BENCHMARK_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE = pathlib.Path(BENCHMARK_DIR) / "run.log"

Logger().configureLoggers(LOG_FILE)
logger = Logger().getLogger()

def generateRandomCircuit():
    inputCirc = random_circuit(CIRC_N_QUBITS, CIRC_DEPTH) # (5,4) most of the time result in a wire cut
    inputCirc.measure_all()
    logger.info(f"random circuit with {CIRC_N_QUBITS} qubits & depth of {CIRC_DEPTH} is generated")
    return inputCirc

def generateSupremacy():
    inputCirc = generate_circ(CIRC_N_QUBITS, CIRC_DEPTH, "supremacy")
    inputCirc.measure_all()
    logger.info(f"supremacy circuit with {CIRC_N_QUBITS} qubits & depth of {CIRC_DEPTH} is generated")
    return inputCirc

def generateEfficientSu2():
    entanglement = "linear"
    inputCirc = EfficientSU2(CIRC_N_QUBITS, entanglement=entanglement, reps=2)
    inputCirc = inputCirc.bind_parameters(
        {param: np.random.randn() / 2 for param in inputCirc.parameters}
    )
    inputCirc.measure_all()
    logger.info(f"EfficientSU2 circuit with {CIRC_N_QUBITS} qubits & {entanglement} entanglement is generated")
    return inputCirc

def generateGhz():
    inputCirc = QuantumCircuit(CIRC_N_QUBITS, CIRC_N_QUBITS)
    inputCirc.h(0)
    for i in range(1, CIRC_N_QUBITS):
        inputCirc.cx(i - 1, i)
    inputCirc.measure_all()
    logger.info(f"linear GHZ state circuit with {CIRC_N_QUBITS} qubits is generated")
    return inputCirc

def generateSycamore():
    inputCirc = generate_circ(CIRC_N_QUBITS, CIRC_DEPTH, "sycamore")
    inputCirc.measure_all()
    logger.info(f"sycamore circuit with {CIRC_N_QUBITS} qubits & depth of {CIRC_DEPTH} is generated")
    return inputCirc

def generateHwea():
    inputCirc = generate_circ(CIRC_N_QUBITS, CIRC_DEPTH, "hwea")
    inputCirc.measure_all()
    logger.info(f"HWEA circuit with {CIRC_N_QUBITS} qubits & depth of {CIRC_DEPTH} is generated")
    return inputCirc

def generateBv():
    inputCirc = generate_circ(CIRC_N_QUBITS, CIRC_DEPTH, "bv")
    inputCirc.measure_all()
    logger.info(f"BV circuit with {CIRC_N_QUBITS} qubits & depth of {CIRC_DEPTH} is generated")
    return inputCirc

def generateQft():
    inputCirc = generate_circ(CIRC_N_QUBITS, CIRC_DEPTH, "qft")
    inputCirc.measure_all()
    logger.info(f"QFT circuit with {CIRC_N_QUBITS} qubits & depth of {CIRC_DEPTH} is generated")
    return inputCirc

def generateAqft():
    inputCirc = generate_circ(CIRC_N_QUBITS, CIRC_DEPTH, "aqft")
    inputCirc.measure_all()
    logger.info(f"AQFT circuit with {CIRC_N_QUBITS} qubits & depth of {CIRC_DEPTH} is generated")
    return inputCirc

def generateAdder():
    inputCirc = generate_circ(CIRC_N_QUBITS, CIRC_DEPTH, "adder")
    inputCirc.measure_all()
    logger.info(f"Adder circuit with {CIRC_N_QUBITS} qubits & depth of {CIRC_DEPTH} is generated")
    return inputCirc

def generateErdos():
    inputCirc = generate_circ(CIRC_N_QUBITS, CIRC_DEPTH, "erdos")
    inputCirc.measure_all()
    logger.info(f"QAOA Erdos circuit with {CIRC_N_QUBITS} qubits & depth of {CIRC_DEPTH} is generated")
    return inputCirc

if CIRC_NAME == "ran":
    inputCirc = generateRandomCircuit()
elif CIRC_NAME == "sup":
    inputCirc = generateSupremacy()
elif CIRC_NAME == "su":
    inputCirc = generateEfficientSu2()
elif CIRC_NAME == "ghz":
    inputCirc = generateGhz()
elif CIRC_NAME == "syc":
    inputCirc = generateSycamore()
elif CIRC_NAME == "hwe":
    inputCirc = generateHwea()
elif CIRC_NAME == "bv":
    inputCirc = generateBv()
elif CIRC_NAME == "qft":
    inputCirc = generateQft()
elif CIRC_NAME == "aqft":
    inputCirc = generateAqft()
elif CIRC_NAME == "add":
    inputCirc = generateAdder()
elif CIRC_NAME == "erd":
    inputCirc = generateErdos()
else:
    raise RuntimeError("CIRC_NAME {CIRC_NAME} is not supported")


cutter = Cutter(inputCirc=inputCirc, maxNPartitions=BENCHMARK_MAX_PARTITIONS, maxNQubitsPerPartition=BENCHMARK_MAX_N_QUBITS, forceNWireCut=None, forceNGateCut=None)

startTime = datetime.datetime.now()
logger.info(f"solving STARTED")

success = cutter.solve()

endTime = datetime.datetime.now()
logger.info(f"solving DONE")
logger.info(f"solving time elapsed: {endTime-startTime}")
logger.info(f"success => {success}")

if not success:
    sys.exit(0)

decomposedCirc, markedCirc, cutCirc = cutter.getCutCirc()
S, nWireCuts, nGateCuts, Q, Q_pArr = cutter.getModelKeyResults()

logger.info(f"S: {S}")
logger.info(f"Q: {Q}")
logger.info(f"nWireCuts: {nWireCuts}")
logger.info(f"nGateCuts: {nGateCuts}")
for idx, Q_pi in enumerate(Q_pArr):
    logger.info(f"Q_p{idx}: {Q_pi}")

cutter.logOptimizerResults()

Utilities.saveCircuit(decomposedCirc, BENCHMARK_DIR, "1_decomposedCirc")
Utilities.saveCircuit(markedCirc, BENCHMARK_DIR, "2_markedCirc")
Utilities.saveCircuit(cutCirc, BENCHMARK_DIR, "3_cutCirc")

nShots = 10000
backend = FakeKolkataV2()

logger.info(f"Circuits will be run with {nShots} shots to calculate fidelity...")

inputCircFidelity, cutCircFidelity, cutVsUncutFidelity = Utilities.compareOriginalCircWithCutCirc(decomposedCirc, cutCirc, backend, nShots)

logger.info(f"inputCircFidelity: {inputCircFidelity}")
logger.info(f"cutCircFidelity: {cutCircFidelity}")
logger.info(f"cutVsUncutFidelity: {cutVsUncutFidelity}")