import threading
from typing import List, Dict, Tuple
import sys

import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.providers import BackendV2
from qiskit.quantum_info import hellinger_fidelity
from qiskit.visualization import dag_drawer
from qiskit_aer import AerSimulator


from qvm.quasi_distr import QuasiDistr
from qvm.run import run_virtual_circuit, generate_instantiations
from qvm.virtual_circuit import VirtualCircuit

from HwAwareCutter.Logger import Logger


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
        Logger().getLogger(__name__).debug(f"getCircResultFromBackend {backend} => STARTED with nShots: {nShots}")
        resultsMap[backend] = QuasiDistr.from_counts(
            backend.run(circuit, shots=nShots).result().get_counts()
        )
        Logger().getLogger(__name__).debug(f"getCircResultFromBackend {backend} => ENDED")

    results = {}
    simulatorBackend = AerSimulator()

    optimizedInputCirc = transpile(circuit, simulatorBackend, optimization_level=3)
    idealSimulatorThread = threading.Thread(target=workerTask, args=[optimizedInputCirc, simulatorBackend, nShots, results])
    optimizedNoisyCirc = transpile(circuit, backend, optimization_level=3)
    noisyBackendThread = threading.Thread(target=workerTask, args=[optimizedNoisyCirc, backend, nShots, results])

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
        Logger().getLogger(__name__).debug(f"getVirtualCircResultFromBackend {backend} STARTED with nShots: {nShots}")
        virtualCirc.set_backend_for_all(backend)
        resultsMap[backend], _ = run_virtual_circuit(virtualCirc, shots=nShots)
        Logger().getLogger(__name__).debug(f"getVirtualCircResultFromBackend {backend} DONE")

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


def getVirtualCircResultFromBackends(cutCircuit: QuantumCircuit, backends: List[BackendV2], nShots : int) -> Tuple[QuasiDistr, QuasiDistr]:
    
    INPUT_KEY = "input"
    CUT_CIRC_KEY = "cut"

    def workerTask1(virtualCirc : VirtualCircuit, backend : BackendV2, nShots : int, resultsMap : Dict[BackendV2, QuasiDistr]) -> None:
        Logger().getLogger(__name__).debug(f"getVirtualCircResultFromBackend {backend} STARTED")
        virtualCirc.set_backend_for_all(backend)
        resultsMap[INPUT_KEY], _ = run_virtual_circuit(virtualCirc, shots=nShots)
        Logger().getLogger(__name__).debug(f"getVirtualCircResultFromBackend {backend} DONE")

    def workerTask2(virtualCirc : VirtualCircuit, backends : List[BackendV2], nShots : int, resultsMap : Dict[BackendV2, QuasiDistr]) -> None:
        Logger().getLogger(__name__).debug(f"getVirtualCircResultFromBackend {backends} STARTED")
        
        frags = virtualCirc.fragment_circuits

        for idx, fKey in enumerate(frags.keys()):
            assert(frags[fKey].num_qubits <= backends[idx].configuration().n_qubits)
            virtualCirc.set_backend(fKey, backends[idx])

        resultsMap[CUT_CIRC_KEY], _ = run_virtual_circuit(virtualCirc, shots=nShots)
        Logger().getLogger(__name__).debug(f"getVirtualCircResultFromBackend {backends} DONE")

    results = {}
    simulatorBackend = AerSimulator()

    idealSimulatorThread = threading.Thread(target=workerTask1, args=[VirtualCircuit(cutCircuit.copy()), simulatorBackend, nShots, results])
    noisyBackendThread = threading.Thread(target=workerTask2, args=[VirtualCircuit(cutCircuit.copy()), backends, nShots, results])

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

    return results[INPUT_KEY], results[CUT_CIRC_KEY]


# return originalCircFidelity, cutCircFidelity, cutVsUncutFidelity
def compareOriginalCircWithCutCirc(originalCirc : QuantumCircuit, cutCirc : QuantumCircuit, backend: BackendV2, nShots : int) -> Tuple[float, float, float]:
    results = {}

    virtualCutCirc = VirtualCircuit(cutCirc.copy())
    nInstances = 0
    for frag, frag_circuit in virtualCutCirc.fragment_circuits.items():
        instance_labels = virtualCutCirc.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circuit, instance_labels)
        nInstances += len(instantiations)

    def originalCircTask(originalCirc : QuantumCircuit, backend : BackendV2, nShots : int, results : List) -> None:
        Logger().getLogger(__name__).debug("originalCircTask STARTED")
        idealResult, noisyResult = getCircResultFromBackend(originalCirc, backend, nShots)
        results[originalCirc.name] = (idealResult, noisyResult)
        Logger().getLogger(__name__).debug("originalCircTask ENDED")

    def cutCircTask(cutCirc : QuantumCircuit, backend : BackendV2, nShots : int, results : List) -> None:
        Logger().getLogger(__name__).debug("cutCircTask STARTED")
        idealResult, noisyResult = getVirtualCircResultFromBackend(cutCirc, backend, nShots)
        results[cutCirc.name] = (idealResult, noisyResult)
        Logger().getLogger(__name__).debug("cutCircTask ENDED")

    originalCircThread = threading.Thread(target=originalCircTask, args=[originalCirc, backend, nInstances*nShots, results])
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

    Logger().getLogger(__name__).debug("inputCircIdealResult: ")
    Logger().getLogger(__name__).debug(dict(sorted(inputCircIdealResult.items())))
    Logger().getLogger(__name__).debug("cutCircIdealResult: ")
    Logger().getLogger(__name__).debug(dict(sorted(cutCircIdealResult.items())))

    inputCircIdealResultKeysSet = set(inputCircIdealResult.keys())
    cutCircIdealResultKeysSet = set(cutCircIdealResult.keys())

    sameKeys = inputCircIdealResultKeysSet.intersection(cutCircIdealResultKeysSet)
    sameKeysCompare = {}
    for key in sameKeys:
        sameKeysCompare[key] = (inputCircIdealResult[key], cutCircIdealResult[key])
    
    Logger().getLogger(__name__).debug("sameKeysCompare: ")
    Logger().getLogger(__name__).debug(dict(sorted(sameKeysCompare.items())))
    
    onlyInInputCircuitKeys = inputCircIdealResultKeysSet.difference(cutCircIdealResultKeysSet)
    onlyInCutCircuitKeys = cutCircIdealResultKeysSet.difference(inputCircIdealResultKeysSet)

    onlyInInputCircuit = {k : inputCircIdealResult[k] for k in onlyInInputCircuitKeys}
    onlyInCutCircuit = {k : cutCircIdealResult[k] for k in onlyInCutCircuitKeys}

    Logger().getLogger(__name__).debug("onlyInInputCircuit: ")
    Logger().getLogger(__name__).debug(dict(sorted(onlyInInputCircuit.items())))
    Logger().getLogger(__name__).debug("onlyInCutCircuit: ")
    Logger().getLogger(__name__).debug(dict(sorted(onlyInCutCircuit.items())))

    inputCircFidelity = hellinger_fidelity(inputCircIdealResult, inputCircNoisyResult)
    cutCircFidelity = hellinger_fidelity(cutCircIdealResult, cutCircNoisyResult)
    cutVsUncutFidelity = hellinger_fidelity(inputCircIdealResult, cutCircIdealResult)

    return inputCircFidelity, cutCircFidelity, cutVsUncutFidelity


# return originalCircFidelity, cutCircFidelity, cutVsUncutFidelity
def compareOriginalCircWithCutCircMultipleBackends(originalCirc : QuantumCircuit, cutCirc : QuantumCircuit, backendForInputCirc: BackendV2, fragmentBackends : List[BackendV2], nShots : int) -> Tuple[float, float, float]:
    results = {}

    def originalCircTask(originalCirc : QuantumCircuit, backendForInputCirc : BackendV2, nShots : int, results : List) -> None:
        Logger().getLogger(__name__).debug("originalCircTask STARTED")
        idealResult, noisyResult = getCircResultFromBackend(originalCirc, backendForInputCirc, nShots)
        results[originalCirc.name] = (idealResult, noisyResult)
        Logger().getLogger(__name__).debug("originalCircTask ENDED")

    def cutCircTask(cutCirc : QuantumCircuit, fragmentBackends : List[BackendV2], nShots : int, results : List) -> None:
        Logger().getLogger(__name__).debug("cutCircTask STARTED")
        idealResult, noisyResult = getVirtualCircResultFromBackends(cutCirc, fragmentBackends, nShots)
        results[cutCirc.name] = (idealResult, noisyResult)
        Logger().getLogger(__name__).debug("cutCircTask ENDED")

    originalCircThread = threading.Thread(target=originalCircTask, args=[originalCirc, backendForInputCirc, nShots, results])
    cutCircThread = threading.Thread(target=cutCircTask, args=[cutCirc, fragmentBackends, nShots, results])

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

    Logger().getLogger(__name__).debug("inputCircIdealResult: ")
    Logger().getLogger(__name__).debug(dict(sorted(inputCircIdealResult.items())))
    Logger().getLogger(__name__).debug("cutCircIdealResult: ")
    Logger().getLogger(__name__).debug(dict(sorted(cutCircIdealResult.items())))

    inputCircIdealResultKeysSet = set(inputCircIdealResult.keys())
    cutCircIdealResultKeysSet = set(cutCircIdealResult.keys())

    sameKeys = inputCircIdealResultKeysSet.intersection(cutCircIdealResultKeysSet)
    sameKeysCompare = {}
    for key in sameKeys:
        sameKeysCompare[key] = (inputCircIdealResult[key], cutCircIdealResult[key])
    
    Logger().getLogger(__name__).debug("sameKeysCompare: ")
    Logger().getLogger(__name__).debug(dict(sorted(sameKeysCompare.items())))
    
    onlyInInputCircuitKeys = inputCircIdealResultKeysSet.difference(cutCircIdealResultKeysSet)
    onlyInCutCircuitKeys = cutCircIdealResultKeysSet.difference(inputCircIdealResultKeysSet)

    onlyInInputCircuit = {k : inputCircIdealResult[k] for k in onlyInInputCircuitKeys}
    onlyInCutCircuit = {k : cutCircIdealResult[k] for k in onlyInCutCircuitKeys}

    Logger().getLogger(__name__).debug("onlyInInputCircuit: ")
    Logger().getLogger(__name__).debug(dict(sorted(onlyInInputCircuit.items())))
    Logger().getLogger(__name__).debug("onlyInCutCircuit: ")
    Logger().getLogger(__name__).debug(dict(sorted(onlyInCutCircuit.items())))

    inputCircFidelity = hellinger_fidelity(inputCircIdealResult, inputCircNoisyResult)
    cutCircFidelity = hellinger_fidelity(cutCircIdealResult, cutCircNoisyResult)
    cutVsUncutFidelity = hellinger_fidelity(inputCircIdealResult, cutCircIdealResult)

    return inputCircFidelity, cutCircFidelity, cutVsUncutFidelity