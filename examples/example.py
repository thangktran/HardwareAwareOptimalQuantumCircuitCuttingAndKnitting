from qiskit.circuit import QuantumCircuit
from qiskit.providers.fake_provider import FakeKolkataV2

from HwAwareCutter.Cutter import Cutter
from HwAwareCutter import Utilities

nQubits = 3
inputCirc = QuantumCircuit(nQubits, nQubits)
inputCirc.cx(0, 1)
inputCirc.cx(0, 2)
inputCirc.h(0)
inputCirc.cx(0, 1)
inputCirc.cx(1, 2)
inputCirc.cx(0, 1)
inputCirc.measure_all()

cutter = Cutter(inputCirc=inputCirc, maxNPartitions=2, maxNQubitsPerPartition=10, forceWireCut=True)

cutter.solve()
decomposedCirc, markedCirc, cutCirc = cutter.getCutCirc()
S, nWireCuts, nGateCuts, Q, Q_pArr = cutter.getModelKeyResuts()

print(f"S: {S}")
print(f"Q: {Q}")
print(f"nWireCuts: {nWireCuts}")
print(f"nGateCuts: {nGateCuts}")
for idx, Q_pi in enumerate(Q_pArr):
    print(f"Q_p{idx}: {Q_pi}")

nShots = 10000
backend = FakeKolkataV2()

inputCircFidelity, cutCircFidelity, idealResultDiff = Utilities.compareOriginalCircWithCutCirc(inputCirc, cutCirc, backend, nShots)

print(f"inputCircFidelity: {inputCircFidelity}")
print(f"cutCircFidelity: {cutCircFidelity}")
print(f"idealResultDiff: {idealResultDiff}")

Utilities.showCircuitsAndDags(circuits=[decomposedCirc, markedCirc, cutCirc], dags=[])