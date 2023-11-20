import pathlib

from qiskit.circuit import QuantumCircuit
from qiskit.providers.fake_provider import FakeKolkataV2

from HwAwareCutter.Cutter import Cutter
from HwAwareCutter import Utilities
from HwAwareCutter.Logger import Logger

PATH = pathlib.Path("./examples/")
Logger().configureLoggers(PATH / pathlib.Path("run.log"))
logger = Logger().getLogger()

nQubits = 3
inputCirc = QuantumCircuit(nQubits, nQubits)
inputCirc.cx(0, 1)
inputCirc.cx(0, 2)
inputCirc.h(0)
inputCirc.cx(0, 1)
inputCirc.cx(1, 2)
inputCirc.cx(0, 1)
inputCirc.measure_all()

cutter = Cutter(inputCirc=inputCirc, maxNPartitions=2, maxNQubitsPerPartition=10, forceNWireCut=1, forceNGateCut=2)

while cutter.solve():

    logger.info(f"success => True")
        
    decomposedCirc, markedCirc, cutCirc = cutter.getCutCirc()
    S, nWireCuts, nGateCuts, Q, Q_pArr = cutter.getModelKeyResults()

    logger.info(f"S: {S}")
    logger.info(f"Q: {Q}")
    logger.info(f"nWireCuts: {nWireCuts}")
    logger.info(f"nGateCuts: {nGateCuts}")
    for idx, Q_pi in enumerate(Q_pArr):
        logger.info(f"Q_p{idx}: {Q_pi}")

    cutter.logOptimizerResults()

    nShots = 10000
    backend = FakeKolkataV2()

    inputCircFidelity, cutCircFidelity, cutVsUncutFidelity = Utilities.compareOriginalCircWithCutCirc(inputCirc, cutCirc, backend, nShots)

    logger.info(f"inputCircFidelity: {inputCircFidelity}")
    logger.info(f"cutCircFidelity: {cutCircFidelity}")
    logger.info(f"cutVsUncutFidelity: {cutVsUncutFidelity}")

    # Utilities.showCircuitsAndDags(circuits=[decomposedCirc, markedCirc, cutCirc], dags=[])
    Utilities.saveCircuit(decomposedCirc, PATH, "1_decomposedCirc")
    Utilities.saveCircuit(markedCirc, PATH, "2_markedCirc")
    Utilities.saveCircuit(cutCirc, PATH, "3_cutCirc")

logger.info("program exiting ...")