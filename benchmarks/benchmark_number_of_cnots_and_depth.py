import pathlib
import sys
import datetime

from qiskit import transpile
from qiskit.providers.fake_provider import FakeKolkataV2

from helper_functions import genCirc

from HwAwareCutter.Cutter import Cutter
from HwAwareCutter.Logger import Logger
from HwAwareCutter import Utilities

from qvm.virtual_circuit import *


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

cutter = Cutter(inputCirc=inputCirc, maxNPartitions=BENCHMARK_MAX_PARTITIONS, maxNQubitsPerPartition=BENCHMARK_MAX_N_QUBITS, forceNWireCuts=None, forceNGateCuts=None, maxNQpdCuts=5, maxNCuts=5, maxCutsPerPartitions=5)

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
S, A, L, nWireCuts, nGateCuts, Q, Q_pArr, C, C_pArr = cutter.getModelKeyResults()

def getParams(c):
    countOps = c.count_ops()
    nCNots = countOps["cx"] if "cx" in countOps else 0
    return nCNots, c.depth()

backend = FakeKolkataV2()

nCnots, depth = getParams(inputCirc)
logger.info(f"PARAM == inputCirc => nCnots: {nCnots}; depth: {depth}")

transpiledInputCirc = transpile(inputCirc, backend)
nCnots, depth = getParams(transpiledInputCirc)
logger.info(f"PARAM == transpiledInputCirc => nCnots: {nCnots}; depth: {depth}")

cutCircInst = []
sumNCnots = 0
depthSet = set()
sumNCnotsInst = 0
depthSetInst = set()
virt = VirtualCircuit(cutCirc.copy())
idx=0

for frag, frag_circuit in virt.fragment_circuits.items():

        transpiledFrag = transpile(frag_circuit, backend)
        nCnots, depth = getParams(transpiledFrag)
        logger.debug(f"PARAM == transpiledFrag[{idx}] => nCnots: {nCnots}; depth: {depth}")
        idx+=1
        sumNCnots += nCnots
        depthSet.add(depth)

        instance_labels = virt.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circuit, instance_labels)
        cutCircInst.extend(instantiations)

logger.info(f"PARAM == transpiledFrag => sumNCnots: {sumNCnots}; max(depthSet): {max(depthSet)}")