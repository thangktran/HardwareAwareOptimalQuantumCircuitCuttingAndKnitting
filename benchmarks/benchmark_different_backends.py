import pathlib
import sys
import datetime

from qiskit.providers.fake_provider import FakeArmonk, FakeOpenPulse2Q, FakeOpenPulse3Q, FakeAthens, FakeJakarta, FakeKolkataV2

from helper_functions import genCirc

from HwAwareCutter.Cutter import Cutter
from HwAwareCutter import Utilities
from HwAwareCutter.Logger import Logger


# FakeArmonk(V2) : 1q
# FakeOpenPulse2Q : 2q
# FakeOpenPulse3Q : 3q
# FakeAthens(V2) : 5q
# FakeJakarta(V2) : 7q
# BACKENDS = [FakeArmonk(), FakeOpenPulse2Q(), FakeOpenPulse3Q(), FakeAthens()]
BACKENDS = [FakeOpenPulse2Q(), FakeOpenPulse3Q(), FakeAthens()]
BENCHMARK_MAX_PARTITIONS = len(BACKENDS)
BENCHMARK_MAX_N_QUBITS = [b.configuration().n_qubits for b in BACKENDS]
BENCHMARK_DIR = ""
CIRC_NAME = "ghz"
CIRC_N_QUBITS = 10
CIRC_DEPTH = 1


BENCHMARK_DIR = f"./benchmark_results/{CIRC_NAME}_{CIRC_N_QUBITS}_{CIRC_DEPTH}_{BENCHMARK_MAX_PARTITIONS}_{BENCHMARK_MAX_N_QUBITS}_{datetime.datetime.now()}"
INSTANTIANTIONS_DIR = f"{BENCHMARK_DIR}/instantiations"
pathlib.Path(INSTANTIANTIONS_DIR).mkdir(parents=True, exist_ok=True)
LOG_FILE = pathlib.Path(BENCHMARK_DIR) / "run.log"

Logger().configureLoggers(LOG_FILE)
logger = Logger().getLogger()

inputCirc = genCirc(CIRC_NAME, CIRC_N_QUBITS, CIRC_DEPTH)

cutter = Cutter(inputCirc=inputCirc, maxNPartitions=BENCHMARK_MAX_PARTITIONS, maxNQubitsPerPartition=BENCHMARK_MAX_N_QUBITS, forceNWireCuts=None, forceNGateCuts=None, maxNCuts=5)

startTime = datetime.datetime.now()
logger.info(f"solving STARTED")

success = cutter.solve()

endTime = datetime.datetime.now()
logger.info(f"solving DONE")
logger.info(f"solving time elapsed: {endTime-startTime}")
logger.info(f"success => {success}")

if not success:
    sys.exit(0)

decomposedCirc, markedCirc, cutCirc, instantiations = cutter.getResultCircs()
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


nShots = 1000
backend = FakeKolkataV2()

logger.info(f"Circuits will be run with {nShots} shots to calculate fidelity...")

inputCircFidelity, cutCircFidelity, cutVsUncutFidelity = Utilities.compareOriginalCircWithCutCircMultipleBackends(decomposedCirc, cutCirc, backend, BACKENDS, nShots)

logger.info(f"inputCircFidelity: {inputCircFidelity}")
logger.info(f"cutCircFidelity: {cutCircFidelity}")
logger.info(f"cutVsUncutFidelity: {cutVsUncutFidelity}")