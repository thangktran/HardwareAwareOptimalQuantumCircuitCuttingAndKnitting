from z3 import *
import numpy as np
import matplotlib.pyplot as plt

from qiskit import *
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.visualization import dag_drawer
from qiskit.circuit import Qubit
from qiskit.circuit.library import Barrier
from qiskit.circuit.random import random_circuit

from dataclasses import dataclass
from enum import Enum
from typing import *
from collections import defaultdict
import math
from random import randrange
import datetime
import sys
import pathlib

from qcg import generators

BENCHMARK_N_PARTITIONS = None
BENCHMARK_RUNNING = False
BENCHMARK_DIR = ""
if len(sys.argv) > 2 and sys.argv[1] == "-p":
    BENCHMARK_RUNNING = True
    BENCHMARK_N_PARTITIONS = int(sys.argv[2])
    BENCHMARK_DIR = ""
    if len(sys.argv) == 4:
        BENCHMARK_DIR = f"./benchmark_result/{sys.argv[3]}/{BENCHMARK_N_PARTITIONS}"
    else:
        BENCHMARK_DIR = f"./benchmark_result/{BENCHMARK_N_PARTITIONS}"
    pathlib.Path(BENCHMARK_DIR).mkdir(parents=True, exist_ok=True)

print(f"start time: {datetime.datetime.now()}")

######################################## circuit ###############################################
def defaultTestCircuit():
    nQubits = 3
    input_circ = QuantumCircuit(nQubits, nQubits)
    input_circ.h(0)
    input_circ.cx(0, 1)
    input_circ.cx(0, 2)
    input_circ.cx(0, 1)
    input_circ.cx(1, 2)
    input_circ.cx(0, 1)
    input_circ.measure(range(nQubits), range(nQubits))
    print(f"test circuit with {nQubits} qubits is generated")
    return input_circ
def generateRandomCircuit():
    nQubits = 10 # randrange(3, 15) # 40
    depth = 10 # randrange(5, 20) # 51
    print(f"random circuit with {nQubits} qubits & depth of {depth} is generated")
    return random_circuit(nQubits, depth)
def generateSupremacy():
    nQubits = 16
    depth = 1
    def factor_int(n):
        nsqrt = math.ceil(math.sqrt(n))
        val = nsqrt
        while 1:
            co_val = int(n / val)
            if val * co_val == n:
                return val, co_val
            else:
                val -= 1
    i, j = factor_int(nQubits)
    assert(abs(i - j) <= 2)
    input_circ = generators.gen_supremacy(i, j, depth * 8)
    print(f"supremacy circuit with {nQubits} qubits & depth of {depth} is generated")
    return input_circ
# input_circ = generateRandomCircuit()
# input_circ = defaultTestCircuit()
input_circ = generateSupremacy()

################################ PREPROCESSING STEPS ############################################

# remove 1-qubit gate & barriers
def circuitSanitizer(c : QuantumCircuit) -> QuantumCircuit:
    dag = circuit_to_dag(c)
    cleanDag = DAGCircuit()
    for qreg in c.qregs:
        cleanDag.add_qreg(qreg)
    for v in dag.topological_op_nodes():
        if v.op.name != "barrier" and len(v.qargs) == 2:
            cleanDag.apply_operation_back(op=v.op, qargs=v.qargs)
    return dag_to_circuit(cleanDag)
@dataclass
class DagVertex:
    idx : int
    qubit : Qubit
    nThGate : int
    opNode : DAGOpNode
    opNodeV0Idx : int
    opNodeV1Idx : int
# return tuple of V, W and G according to paper.
# circuit MUST only contains 2 qubit gates.
def readCirc(circuit : QuantumCircuit) -> Tuple[List[DagVertex], List, List, List[DagVertex]]:
    # V contains all vertices.
    # entry is DagVertex
    V: List[DagVertex] = []
    # W contains all possible wire cut edges
    # tuples of (v_1_idx, v_2_idx) where v_n_idx belongs to V
    W = []
    # G contains all possible gate cut edges
    # tuples of (v_1_idx, v_2_idx) where v_n_idx belongs to V
    G = []
    # I contains first vertex of each qubit. I is subset of V.
    I: List[DagVertex] = []
    dag = circuit_to_dag(circuit, False)
    qubitGateCounter = {}
    qubitPreviousVertexIdx = {}
    for qubit in dag.qubits:
        qubitGateCounter[qubit] = 0
        qubitPreviousVertexIdx[qubit] = None
    for v in dag.topological_op_nodes():
        if len(v.qargs) != 2:
            raise Exception("circuit MUST only contains 2 qubit gates")
        qubit0 = v.qargs[0]
        qubit1 = v.qargs[1]
        assert(len(circ.find_bit(qubit0).registers)==1)
        assert(len(circ.find_bit(qubit1).registers)==1)
        v0Idx = len(V)
        v1Idx = v0Idx + 1
        v.op.label = f"{v0Idx}_{v1Idx}"
        # add 2 vertices
        v0 = DagVertex(v0Idx, qubit0, qubitGateCounter[qubit0], v, v0Idx, v1Idx)
        v1 = DagVertex(v1Idx, qubit1, qubitGateCounter[qubit1], v, v0Idx, v1Idx)
        V.append(v0)
        V.append(v1)
        qubitGateCounter[qubit0] += 1
        qubitGateCounter[qubit1] += 1
        # 2 vertices from Op node create a Gate Cut edge
        G.append((v0Idx, v1Idx))
        # check if these qubits was encountered before.
        # if yes, the previous vertex and current vertex of
        # this qubit create a Wire Cut edge.
        if qubitPreviousVertexIdx[qubit0] is not None:
            W.append((qubitPreviousVertexIdx[qubit0], v0Idx))
        else:
            I.append(v0)
        if qubitPreviousVertexIdx[qubit1] is not None:
            W.append((qubitPreviousVertexIdx[qubit1], v1Idx))
        else:
            I.append(v1)
        # update vertex ID for the qubit.
        qubitPreviousVertexIdx[qubit0] = v0Idx
        qubitPreviousVertexIdx[qubit1] = v1Idx
    return V, W, G, I
def checkGraph(vertices, edges):
    nVertices = len(vertices)
    verticesSet = set()
    for (u, v) in edges:
        assert(u < v)
        assert(u < nVertices)
        verticesSet.add(u)
        verticesSet.add(v)
    assert(verticesSet == set(range(nVertices)))
# convert to 2-qubit gates
circ = input_circ.decompose()
# remove 1-qubit gate & barries
circ = circuitSanitizer(circ)
# get V, W, G, I
V, W, G, I= readCirc(circ)
# verify the vertices and edges are correct.
checkGraph(V, W+G)

################################ MODEL VARIABLES ############################################
# number of partitions
N_PARTITIONS = 2 if BENCHMARK_N_PARTITIONS is None else BENCHMARK_N_PARTITIONS
MAX_N_QUBIT_PER_PARTITION = 100
print(f"N_PARTITIONS {N_PARTITIONS}")
print(f"MAX_N_QUBIT_PER_PARTITION {MAX_N_QUBIT_PER_PARTITION}")
# gate qubit vertex v assigned to partition p
o_vp = []
# circuit is cut at edge e
c_e = []
# circuit is simultanously cut at edge e
b_e = []
# total number of qubits in partition p
Q_p = []
# Q is maximum number of qubit per partition
Q = Int('Q')
# S is the total cost of cutting (overhead sampling)
S = Int('S')
# helper object to speed up look-up.
# o_var_lookup[vIdx][pIdx] return the z3 variable.
o_var_lookup = defaultdict(dict)
# populate o_vp
for vIdx in range(len(V)):
    for pIdx in range(N_PARTITIONS):
        variableName = f"o_{vIdx}_{pIdx}"
        var = Bool(variableName)
        # z3.Bool doesn't have these properties
        # we add them to make it simpler to retrieve.
        var.vIdx = vIdx
        var.pIdx = pIdx
        o_vp.append(var)
        o_var_lookup[vIdx][pIdx] = var
# populate c_e
class EdgeType(Enum):
    GateCut = 0
    WireCut = 1
for eIdx in range(len(W)):
    variableName = f"c_{eIdx}[W]_{W[eIdx][0]}_{W[eIdx][1]}"
    var = Bool(variableName)
    # z3.Bool doesn't have these properties
    # we add them to make it simpler to retrieve.
    var.eIdx = eIdx
    var.edge = W[eIdx]
    var.edgeType = EdgeType.WireCut
    c_e.append(var)
for eIdx in range(len(G)):
    variableName = f"c_{eIdx}[G]_{G[eIdx][0]}_{G[eIdx][1]}"
    var = Bool(variableName)
    # z3.Bool doesn't have these properties
    # we add them to make it simpler to retrieve.
    var.eIdx = eIdx
    var.edge = G[eIdx]
    var.edgeType = EdgeType.GateCut
    c_e.append(var)
# populate b_e
for eIdx in range(len(W)):
    variableName = f"b_{eIdx}[W]_{W[eIdx][0]}_{W[eIdx][1]}"
    var = Bool(variableName)
    # z3.Bool doesn't have these properties
    # we add them to make it simpler to retrieve.
    var.eIdx = eIdx
    var.edge = W[eIdx]
    var.edgeType = EdgeType.WireCut
    b_e.append(var)
for eIdx in range(len(G)):
    variableName = f"b_{eIdx}[G]_{G[eIdx][0]}_{G[eIdx][1]}"
    var = Bool(variableName)
    # z3.Bool doesn't have these properties
    # we add them to make it simpler to retrieve.
    var.eIdx = eIdx
    var.edge = G[eIdx]
    var.edgeType = EdgeType.GateCut
    b_e.append(var)
assert(N_PARTITIONS <= len(V))
# populate Q_p
for pIdx in range(N_PARTITIONS):
    variableName = f"Q_p{pIdx}"
    var = Int(variableName)
    # z3.Int doesn't have these properties
    # we add them to make it simpler to retrieve.
    var.pIdx = pIdx
    Q_p.append(var)



################################ MODEL CONSTRAINTS ############################################
s = Optimize()
# c_e constraints
for var in c_e:
    u = var.edge[0]
    v = var.edge[1]
    constraints = [o_var_lookup[u][p]!=o_var_lookup[v][p] for p in range(N_PARTITIONS)]
    s.add(var == Or(constraints))
# o_vp 1st constraints: no vertex is assigned twice
for vIdx in range(len(V)):
    constraints = [ If(i == j,
                       True,
                       Implies( o_var_lookup[vIdx][i], Not(o_var_lookup[vIdx][j]) )
                    )
                    for i in range(N_PARTITIONS) for j in range(N_PARTITIONS)]
    s.add(constraints)
# o_vp 2nd constraints: at least 1 partition is assigned to each vertex v
for vIdx, pIdxs in o_var_lookup.items():
    variables = []
    for pIdx, var in pIdxs.items():
        variables.append(var)
    s.add(Or(variables))
# b_e imply c_e
s.add([Implies(b_e[idx], c_e[idx]) for idx in range(len(b_e))])
# Q_p constraints
for pIdx in range(N_PARTITIONS):
    firstSumTerm = []
    secondSumTerm = []
    thirdSumTerm = []
    # first sum term: all o_vp with v in I
    for v in I:
        vIdx = v.idx
        var = o_var_lookup[vIdx][pIdx]
        firstSumTerm.append(If(var, 1, 0))
    # second sum term
    for c_eVar in c_e:
        # skip all GateCut edge since we only want wire cut edge
        if c_eVar.edgeType == EdgeType.GateCut: # belong to G
            continue
        u, v = c_eVar.edge
        o_vpVar = o_var_lookup[v][pIdx]
        secondSumTerm.append(If(And(c_eVar, o_vpVar), 1, 0))
    # third sum term
    for b_eVar in b_e:
        u, v = b_eVar.edge
        o_vpVar = o_var_lookup[v][pIdx]
        o_upVar = o_var_lookup[u][pIdx]
        thirdSumTerm.append(If(And(b_eVar, Or(o_vpVar, o_upVar)), 1, 0))
    s.add(Q_p[pIdx] == Sum(firstSumTerm+secondSumTerm+thirdSumTerm))
GATE_CUT_COST = 6
WIRE_CUT_COST = 8
productTerm = 1
for idx in range(len(c_e)):
    # TODO: check whether this product term is correct.
    productTerm *= If(And(c_e[idx], c_e[idx].edgeType == EdgeType.GateCut), GATE_CUT_COST, 1) * If(And(c_e[idx], c_e[idx].edgeType == EdgeType.WireCut), WIRE_CUT_COST, 1)
s.add(S == productTerm)
s.add(S>1)
s.add(Q <= MAX_N_QUBIT_PER_PARTITION) # number of qubit <= maximum number of qubit allowed
for pIdx in range(N_PARTITIONS):
    s.add(Q >= Q_p[pIdx])
s.minimize(Q)
s.minimize(S)
# helper constraints
nWireCuts = Int('nWireCuts')
nGateCuts = Int('nGateCuts')
sumWireCuts = [If(And(c_e[idx], c_e[idx].edgeType == EdgeType.WireCut), 1, 0) for idx in range(len(c_e))]
sumGateCuts = [If(And(c_e[idx], c_e[idx].edgeType == EdgeType.GateCut), 1, 0) for idx in range(len(c_e))]
s.add(nWireCuts == Sum(sumWireCuts))
s.add(nGateCuts == Sum(sumGateCuts))

################################ GET MODEL AND PROCESS THEM ############################################
##### HELPER FUNCTION
def printModel(m):
    intStr = ""
    boolStr = ""
    for t in m.decls():
        if is_int(m[t]):
            intStr += f"{t} = {m[t]}\n"
        elif is_true(m[t]):
            boolStr += f"{t} = {m[t]}\n"
    print(intStr)
    print(boolStr)
def outputCircuitPic(circuits, drawDagOfCircuits = False, dags = []):
    for c in circuits:
        c.draw(output='mpl')
        if drawDagOfCircuits:
            dag = circuit_to_dag(c)
            img = dag_drawer(dag=dag, scale=2)
            plt.figure()
            plt.imshow(img)
    for dag in dags:
        img = dag_drawer(dag=dag, scale=2)
        plt.figure()
        plt.imshow(img)
    plt.show()
def saveCircuit(circ, dir, name):
    circ.draw(output='mpl', filename=f"{dir}/{name}")

# TODO: there're multiple models. how to handle them?!
modelStatus = s.check()
print(f"finish time: {datetime.datetime.now()}")
print(f"{modelStatus}\n")
if modelStatus != sat:
    print(f"model is not satisfied. Exiting ...")
    exit(0)
m = s.model()
# replace cut with barriers
resultDag = circuit_to_dag(circ)
for c_eVar in c_e:
    if not is_true(m[c_eVar]):
        continue
    uIdx, vIdx = c_eVar.edge
    u = V[uIdx]
    v = V[vIdx]
    if c_eVar.edgeType == EdgeType.GateCut:
        # TODO: use GateCut(Barrier) Op instead of Barrier directly ?!
        resultDag.substitute_node(v.opNode, Barrier(2, f"{str(c_eVar)}"))
    elif c_eVar.edgeType == EdgeType.WireCut:
        newDag = DAGCircuit()
        newDag.add_qubits(u.opNode.qargs)
        newDag.apply_operation_back(op=u.opNode.op, qargs=u.opNode.qargs)
        newDag.apply_operation_back(op=Barrier(1, f"{str(c_eVar)}"), qargs=[u.qubit])
        newNodesMap = resultDag.substitute_node_with_dag(u.opNode, newDag)
        newNode = [*newNodesMap.values()][0]
        u0 = V[u.opNodeV0Idx]
        u1 = V[u.opNodeV1Idx]
        assert(u.opNode == u0.opNode == u1.opNode)
        assert(newNode.op.name == u.opNode.op.name == u0.opNode.op.name == u1.opNode.op.name)
        assert(newNode.op.label == u.opNode.op.label == u0.opNode.op.label == u1.opNode.op.label)
        assert(newNode.qargs == u.opNode.qargs == u0.opNode.qargs == u1.opNode.qargs)
        u0.opNode = newNode
        u1.opNode = newNode
        
resultCirc = dag_to_circuit(resultDag)

################# MAIN ###################
circuitsToDraw = [circ, resultCirc]
circuitsToDrawDags = [circ, resultCirc]
printModel(m)
if BENCHMARK_RUNNING:
    saveCircuit(circ, BENCHMARK_DIR, "original")
    saveCircuit(resultCirc, BENCHMARK_DIR, "cutted")
else:
    outputCircuitPic(circuitsToDraw)