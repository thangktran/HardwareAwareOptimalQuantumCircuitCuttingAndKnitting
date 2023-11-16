import threading
from typing import List, Dict, Tuple
import sys

import matplotlib.pyplot as plt

from qiskit.circuit import Qubit, QuantumCircuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.providers import BackendV2
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator


from qvm.quasi_distr import QuasiDistr
from qvm.run import run_virtual_circuit
from qvm.virtual_circuit import VirtualCircuit


def showCircuitsAndDags(circuits : List[QuantumCircuit] = [], dags : List[DAGCircuit]= []) -> None:
    for c in circuits:
        c.draw(output='mpl')
    for dag in dags:
        img = dag_drawer(dag=dag, scale=2)
        plt.figure()
        plt.imshow(img)
    plt.show()


def saveCircuit(circ : QuantumCircuit, dir : str, name : str) -> None:
    circ.draw(output='mpl', filename=f"{dir}/{name}")


# return tuple of QuasiDistr: (simulatorQuasiDistr, backendQuasiDistr)
# this runs `circuit` on 2 backend simultanously. 1 backend is `AerSimulator`, the
# other one is specified by `backend`.
def getCircResultFromBackend(circuit: QuantumCircuit, backend: BackendV2, nShots : int) -> Tuple[QuasiDistr, QuasiDistr]:
    
    def workerTask(circuit : QuantumCircuit, backend : BackendV2, nShots : int, resultsMap : Dict[BackendV2, QuasiDistr]) -> None:
        print(f"getCircResultFromBackend {backend}")
        resultsMap[backend] = QuasiDistr.from_counts(
            backend.run(circuit, shots=nShots).result().get_counts()
        )

    results = {}
    simulatorBackend = AerSimulator()

    idealSimulatorThread = threading.Thread(target=workerTask, args=[circuit, simulatorBackend, nShots, results])
    noisyBackendThread = threading.Thread(target=workerTask, args=[circuit, backend, nShots, results])

    idealSimulatorThread.start()
    noisyBackendThread.start()

    try:
        while idealSimulatorThread.is_alive() or noisyBackendThread.is_alive():
            if idealSimulatorThread.is_alive():
                idealSimulatorThread.join(0.5)
            if noisyBackendThread.is_alive():
                noisyBackendThread.join(0.5)
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)

    idealSimulatorThread.join()
    noisyBackendThread.join()

    return results[simulatorBackend], results[backend]


# same as getCircResultFromBackend(), but this function
# will turn circuit into a virtualCirc and run it.
def getVirtualCircResultFromBackend(cutCircuit: QuantumCircuit, backend: BackendV2, nShots : int) -> Tuple[QuasiDistr, QuasiDistr]:
    
    def workerTask(virtualCirc : VirtualCircuit, backend : BackendV2, nShots : int, resultsMap : Dict[BackendV2, QuasiDistr]) -> None:
        print(f"getVirtualCircResultFromBackend {backend}")
        virtualCirc.set_backend_for_all(backend)
        resultsMap[backend], _ = run_virtual_circuit(virtualCirc, shots=nShots)

    results = {}
    simulatorBackend = AerSimulator()

    idealSimulatorThread = threading.Thread(target=workerTask, args=[VirtualCircuit(cutCircuit.copy()), simulatorBackend, nShots, results])
    noisyBackendThread = threading.Thread(target=workerTask, args=[VirtualCircuit(cutCircuit.copy()), backend, nShots, results])

    idealSimulatorThread.start()
    noisyBackendThread.start()

    try:
        while idealSimulatorThread.is_alive() or noisyBackendThread.is_alive():
            if idealSimulatorThread.is_alive():
                idealSimulatorThread.join(0.5)
            if noisyBackendThread.is_alive():
                noisyBackendThread.join(0.5)
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)

    idealSimulatorThread.join()
    noisyBackendThread.join()

    return results[simulatorBackend], results[backend]


# return originalCircFidelity, cutCircFidelity, idealResultDiff
def compareOriginalCircWithCutCirc(originalCirc : QuantumCircuit, cutCirc : QuantumCircuit, backend: BackendV2, nShots : int) -> Tuple[float, float, float]:
    results = {}

    def originalCircTask(originalCirc : QuantumCircuit, backend : BackendV2, nShots : int, results : List) -> None:
        print("originalCircTask")
        idealResult, noisyResult = getCircResultFromBackend(originalCirc, backend, nShots)
        results[originalCirc.name] = (idealResult, noisyResult)

    def cutCircTask(cutCirc : QuantumCircuit, backend : BackendV2, nShots : int, results : List) -> None:
        print("cutCircTask")
        idealResult, noisyResult = getVirtualCircResultFromBackend(cutCirc, backend, nShots)
        results[cutCirc.name] = (idealResult, noisyResult)

    originalCircThread = threading.Thread(target=originalCircTask, args=[originalCirc, backend, nShots, results])
    cutCircThread = threading.Thread(target=cutCircTask, args=[cutCirc, backend, nShots, results])

    originalCircThread.start()
    cutCircThread.start()

    try:
        while originalCircThread.is_alive() or cutCircThread.is_alive():
            if originalCircThread.is_alive():
                originalCircThread.join(0.5)
            if cutCircThread.is_alive():
                cutCircThread.join(0.5)
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)

    originalCircThread.join()
    cutCircThread.join()

    inputCircIdealResult, inputCircNoisyResult = results[originalCirc.name]
    cutCircIdealResult, cutCircNoisyResult = results[cutCirc.name]

    inputCircFidelity = hellinger_fidelity(inputCircIdealResult, inputCircNoisyResult)
    cutCircFidelity = hellinger_fidelity(cutCircIdealResult, cutCircNoisyResult)
    idealResultDiff = hellinger_fidelity(inputCircIdealResult, cutCircIdealResult)

    return inputCircFidelity, cutCircFidelity, idealResultDiff