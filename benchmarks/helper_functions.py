# credit: helper_functions/benchmarks.py from cutqc

import qiskit.circuit.library as library
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import EfficientSU2
from qiskit import QuantumCircuit

import math, qiskit, random
import networkx as nx
import numpy as np

from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_sycamore, gen_adder

from HwAwareCutter.Logger import Logger

def factor_int(n):
    nsqrt = math.ceil(math.sqrt(n))
    val = nsqrt
    while 1:
        co_val = int(n / val)
        if val * co_val == n:
            return val, co_val
        else:
            val -= 1

def gen_secret(num_qubit):
    num_digit = num_qubit - 1
    num = 2**num_digit - 1
    num = bin(num)[2:]
    num_with_zeros = str(num).zfill(num_digit)
    return num_with_zeros


def construct_qaoa_plus(P, G, params, reg_name, barriers=False, measure=False):
    assert len(params) == 2 * P, "Number of parameters should be 2P"

    nq = len(G.nodes())
    circ = qiskit.QuantumCircuit(qiskit.QuantumRegister(nq, reg_name))

    # Initial state
    circ.h(range(nq))

    gammas = [param for i, param in enumerate(params) if i % 2 == 0]
    betas = [param for i, param in enumerate(params) if i % 2 == 1]
    for i in range(P):
        # Phase Separator Unitary
        for edge in G.edges():
            q_i, q_j = edge
            circ.rz(gammas[i] / 2, [q_i, q_j])
            circ.cx(q_i, q_j)
            circ.rz(-1 * gammas[i] / 2, q_j)
            circ.cx(q_i, q_j)
            if barriers:
                circ.barrier()

        # Mixing Unitary
        for q_i in range(nq):
            circ.rx(-2 * betas[i], q_i)

    if measure:
        circ.measure_all()

    return circ


def generate_circ(num_qubits, depth, circuit_type, reg_name="q", connected_only=False, seed=None):
    random.seed(seed)
    full_circ = None
    num_trials = 100
    density = 0.001
    while num_trials:
        if circuit_type == "supremacy":
            i, j = factor_int(num_qubits)
            if abs(i - j) <= 2:
                full_circ = gen_supremacy(i, j, depth * 8, regname=reg_name)
        elif circuit_type == "sycamore":
            i, j = factor_int(num_qubits)
            full_circ = gen_sycamore(i, j, depth, regname=reg_name)
        elif circuit_type == "hwea":
            full_circ = gen_hwea(num_qubits, depth, regname=reg_name)
        elif circuit_type == "bv":
            full_circ = gen_BV(gen_secret(num_qubits), barriers=False, regname=reg_name)
        elif circuit_type == "qft":
            full_circ = library.QFT(
                num_qubits=num_qubits, approximation_degree=0, do_swaps=False
            ).decompose()
        elif circuit_type == "aqft":
            approximation_degree = int(math.log(num_qubits, 2) + 2)
            full_circ = library.QFT(
                num_qubits=num_qubits,
                approximation_degree=num_qubits - approximation_degree,
                do_swaps=False,
            ).decompose()
        elif circuit_type == "adder":
            full_circ = gen_adder(
                nbits=int((num_qubits - 2) / 2), barriers=False, regname=reg_name
            )
        elif circuit_type == "regular":
            if 3 * num_qubits % 2 == 0:
                graph = nx.random_regular_graph(3, num_qubits)
                full_circ = construct_qaoa_plus(
                    P=depth,
                    G=graph,
                    params=[np.random.uniform(-np.pi, np.pi) for _ in range(2 * depth)],
                    reg_name=reg_name,
                )
        elif circuit_type == "erdos":
            graph = nx.generators.random_graphs.erdos_renyi_graph(num_qubits, density)
            full_circ = construct_qaoa_plus(
                P=depth,
                G=graph,
                params=[np.random.uniform(-np.pi, np.pi) for _ in range(2 * depth)],
                reg_name=reg_name,
            )
            density += 0.001
        else:
            raise Exception("Illegal circuit type:", circuit_type)

        if full_circ is not None and full_circ.num_tensor_factors() == 1:
            break
        elif full_circ is not None and not connected_only:
            break
        else:
            full_circ = None
            num_trials -= 1
    assert full_circ is None or full_circ.num_qubits == num_qubits
    return full_circ


#################################################

def generateRandomCircuit(nQubits : int, depth : int):
    inputCirc = random_circuit(nQubits, depth) # (5,4) most of the time result in a wire cut
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"random circuit with {nQubits} qubits & depth of {depth} is generated")
    return inputCirc

def generateSupremacy(nQubits : int, depth : int):
    inputCirc = generate_circ(nQubits, depth, "supremacy")
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"supremacy circuit with {nQubits} qubits & depth of {depth} is generated")
    return inputCirc

def generateEfficientSu2(nQubits : int, depth : int):
    entanglement = "linear"
    inputCirc = EfficientSU2(nQubits, entanglement=entanglement, reps=2)
    inputCirc = inputCirc.bind_parameters(
        {param: np.random.randn() / 2 for param in inputCirc.parameters}
    )
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"EfficientSU2 circuit with {nQubits} qubits & {entanglement} entanglement is generated")
    return inputCirc

def generateGhz(nQubits : int, depth : int):
    inputCirc = QuantumCircuit(nQubits, nQubits)
    inputCirc.h(0)
    for i in range(1, nQubits):
        inputCirc.cx(i - 1, i)
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"linear GHZ state circuit with {nQubits} qubits is generated")
    return inputCirc

def generateSycamore(nQubits : int, depth : int):
    inputCirc = generate_circ(nQubits, depth, "sycamore")
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"sycamore circuit with {nQubits} qubits & depth of {depth} is generated")
    return inputCirc

def generateHwea(nQubits : int, depth : int):
    inputCirc = generate_circ(nQubits, depth, "hwea")
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"HWEA circuit with {nQubits} qubits & depth of {depth} is generated")
    return inputCirc

def generateBv(nQubits : int, depth : int):
    inputCirc = generate_circ(nQubits, depth, "bv")
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"BV circuit with {nQubits} qubits & depth of {depth} is generated")
    return inputCirc

def generateQft(nQubits : int, depth : int):
    inputCirc = generate_circ(nQubits, depth, "qft")
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"QFT circuit with {nQubits} qubits & depth of {depth} is generated")
    return inputCirc

def generateAqft(nQubits : int, depth : int):
    inputCirc = generate_circ(nQubits, depth, "aqft")
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"AQFT circuit with {nQubits} qubits & depth of {depth} is generated")
    return inputCirc

def generateAdder(nQubits : int, depth : int):
    inputCirc = generate_circ(nQubits, depth, "adder")
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"Adder circuit with {nQubits} qubits & depth of {depth} is generated")
    return inputCirc

def generateErdos(nQubits : int, depth : int):
    inputCirc = generate_circ(nQubits, depth, "erdos")
    inputCirc.measure_all()
    Logger().getLogger(__name__).info(f"QAOA Erdos circuit with {nQubits} qubits & depth of {depth} is generated")
    return inputCirc


def genCirc(circName : str, nQubits : int, depth : int):
    inputCirc = None

    if circName == "ran":
        inputCirc = generateRandomCircuit(nQubits, depth)
    elif circName == "sup":
        inputCirc = generateSupremacy(nQubits, depth)
    elif circName == "su":
        inputCirc = generateEfficientSu2(nQubits, depth)
    elif circName == "ghz":
        inputCirc = generateGhz(nQubits, depth)
    elif circName == "syc":
        inputCirc = generateSycamore(nQubits, depth)
    elif circName == "hwe":
        inputCirc = generateHwea(nQubits, depth)
    elif circName == "bv":
        inputCirc = generateBv(nQubits, depth)
    elif circName == "qft":
        inputCirc = generateQft(nQubits, depth)
    elif circName == "aqft":
        inputCirc = generateAqft(nQubits, depth)
    elif circName == "add":
        inputCirc = generateAdder(nQubits, depth)
    elif circName == "erd":
        inputCirc = generateErdos(nQubits, depth)
    else:
        raise RuntimeError(f"circName {circName} is not supported")
    
    return inputCirc