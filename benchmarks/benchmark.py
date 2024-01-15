import pathlib
import sys
import datetime

from qiskit.providers.fake_provider import FakeKolkataV2

from helper_functions import genCirc

from HwAwareCutter.Cutter import Cutter
from HwAwareCutter import Utilities
from HwAwareCutter.Logger import Logger


BENCHMARK_MAX_PARTITIONS = 2
BENCHMARK_MAX_N_QUBITS = 10
BENCHMARK_DIR = ""
CIRC_NAME = "ghz"
CIRC_N_QUBITS = 5
CIRC_DEPTH = 1
CUT_ONLY = True # don't do fidelity comparision

# usage: python benchmark.py -p 2 -q 10 [ran|sup|su|ghz|syc|hwe|bv|qft|aqft|add|erd] <nQubit> <nDepth>
# su doesn't need nDepth parameter, but a dummy 0 should be use
if len(sys.argv) == 8 and sys.argv[1] == "-p" and sys.argv[3] == "-q":
    BENCHMARK_MAX_PARTITIONS = int(sys.argv[2])
    BENCHMARK_MAX_N_QUBITS = int(sys.argv[4])
    CIRC_NAME = str(sys.argv[5]).lower()
    CIRC_N_QUBITS = int(sys.argv[6])
    CIRC_DEPTH = int(sys.argv[7])

BENCHMARK_DIR = f"./benchmark_results/{CIRC_NAME}_{CIRC_N_QUBITS}_{CIRC_DEPTH}_{BENCHMARK_MAX_PARTITIONS}_{BENCHMARK_MAX_N_QUBITS}_{datetime.datetime.now()}"
INSTANTIANTIONS_DIR = f"{BENCHMARK_DIR}/instantiations"
pathlib.Path(INSTANTIANTIONS_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE = pathlib.Path(BENCHMARK_DIR) / "run.log"

Logger().configureLoggers(LOG_FILE)
logger = Logger().getLogger()

inputCirc = genCirc(CIRC_NAME, CIRC_N_QUBITS, CIRC_DEPTH)

cutter = Cutter(inputCirc=inputCirc, maxNPartitions=BENCHMARK_MAX_PARTITIONS, maxNQubitsPerPartition=BENCHMARK_MAX_N_QUBITS, forceNWireCuts=None, forceNGateCuts=None, maxNQpdCuts=2, maxNCuts=5)

startTime = datetime.datetime.now()
logger.info(f"solving STARTED")

success = cutter.solve()

endTime = datetime.datetime.now()
logger.info(f"solving DONE")
logger.info(f"solving time elapsed: {endTime-startTime}")
logger.info(f"success => {success}")

if not success:
    sys.exit(0)

decomposedCirc, markedCirc, markedCircWithVirtualMoves, cutCirc, instantiations = cutter.getResultCircs(getInstantiations=False)
S, A, L, nWireCuts, nGateCuts, Q, Q_pArr = cutter.getModelKeyResults()

logger.info(f"S: {S}")
logger.info(f"A: {A}")
logger.info(f"L: {L}")
logger.info(f"Q: {Q}")
logger.info(f"nWireCuts: {nWireCuts}")
logger.info(f"nGateCuts: {nGateCuts}")
for idx, Q_pi in enumerate(Q_pArr):
    logger.info(f"Q_p{idx}: {Q_pi}")

cutter.logOptimizerResults()

Utilities.saveCircuit(decomposedCirc, BENCHMARK_DIR, "1_decomposedCirc")
Utilities.saveCircuit(markedCirc, BENCHMARK_DIR, "2_markedCirc")
Utilities.saveCircuit(markedCircWithVirtualMoves, BENCHMARK_DIR, "3_markedCircWithVirtualMoves")
Utilities.saveCircuit(cutCirc, BENCHMARK_DIR, "4_cutCirc")

logger.info(f"all instantiations will be saved to disk ...")

instantiationCount = 0
for fIdx, inst in enumerate(instantiations):
    for cIdx, c in enumerate(inst):
        Utilities.saveCircuit(c, INSTANTIANTIONS_DIR, f"{fIdx}_{cIdx}")
        instantiationCount+=1

logger.info(f"{instantiationCount} instantiations are saved to disk")

if CUT_ONLY:
    logger.info("CUT_ONLY == True => Simulation will not run.")
    sys.exit(0)

nShots = 1000
backend = FakeKolkataV2()

logger.info(f"Circuits will be run with {nShots} shots to calculate fidelity...")

inputCircFidelity, cutCircFidelity, cutVsUncutFidelity = Utilities.compareOriginalCircWithCutCirc(decomposedCirc, cutCirc, backend, nShots)

logger.info(f"inputCircFidelity: {inputCircFidelity}")
logger.info(f"cutCircFidelity: {cutCircFidelity}")
logger.info(f"cutVsUncutFidelity: {cutVsUncutFidelity}")