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
def getVirtualCircResultFromBackend(cuttedCircuit: QuantumCircuit, backend: BackendV2, nShots : int) -> Tuple[QuasiDistr, QuasiDistr]:
    
    def workerTask(virtualCirc : VirtualCircuit, backend : BackendV2, nShots : int, resultsMap : Dict[BackendV2, QuasiDistr]) -> None:
        print(f"getVirtualCircResultFromBackend {backend}")
        virtualCirc.set_backend_for_all(backend)
        resultsMap[backend], _ = run_virtual_circuit(virtualCirc, shots=nShots)

    results = {}
    simulatorBackend = AerSimulator()

    idealSimulatorThread = threading.Thread(target=workerTask, args=[VirtualCircuit(cuttedCircuit.copy()), simulatorBackend, nShots, results])
    noisyBackendThread = threading.Thread(target=workerTask, args=[VirtualCircuit(cuttedCircuit.copy()), backend, nShots, results])

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


# return originalCircFidelity, cuttedCircFidelity, idealResultDiff
def compareOriginalCircWithCuttedCirc(originalCirc : QuantumCircuit, cuttedCirc : QuantumCircuit, backend: BackendV2, nShots : int) -> Tuple[float, float, float]:
    results = {}

    def originalCircTask(originalCirc : QuantumCircuit, backend : BackendV2, nShots : int, results : List) -> None:
        print("originalCircTask")
        idealResult, noisyResult = getCircResultFromBackend(originalCirc, backend, nShots)
        results[originalCirc.name] = (idealResult, noisyResult)

    def cuttedCircTask(cuttedCirc : QuantumCircuit, backend : BackendV2, nShots : int, results : List) -> None:
        print("cuttedCircTask")
        idealResult, noisyResult = getVirtualCircResultFromBackend(cuttedCirc, backend, nShots)
        results[cuttedCirc.name] = (idealResult, noisyResult)

    originalCircThread = threading.Thread(target=originalCircTask, args=[originalCirc, backend, nShots, results])
    cuttedCircThread = threading.Thread(target=cuttedCircTask, args=[cuttedCirc, backend, nShots, results])

    originalCircThread.start()
    cuttedCircThread.start()

    try:
        while originalCircThread.is_alive() or cuttedCircThread.is_alive():
            if originalCircThread.is_alive():
                originalCircThread.join(0.5)
            if cuttedCircThread.is_alive():
                cuttedCircThread.join(0.5)
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)

    originalCircThread.join()
    cuttedCircThread.join()

    inputCircIdealResult, inputCircNoisyResult = results[originalCirc.name]
    cuttedCircIdealResult, cuttedCircNoisyResult = results[cuttedCirc.name]

    inputCircFidelity = hellinger_fidelity(inputCircIdealResult, inputCircNoisyResult)
    cuttedCircFidelity = hellinger_fidelity(cuttedCircIdealResult, cuttedCircNoisyResult)
    idealResultDiff = hellinger_fidelity(inputCircIdealResult, cuttedCircIdealResult)

    return inputCircFidelity, cuttedCircFidelity, idealResultDiff