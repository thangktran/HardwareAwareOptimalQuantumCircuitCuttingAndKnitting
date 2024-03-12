from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import *

from clingo.solving import Symbol
from clingo.control import Control
import networkx as nx

from qiskit import QuantumRegister
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.circuit.library import SwapGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.dagcircuit.dagcircuit import DAGCircuit

from qvm.compiler.dag import DAG
from qvm.virtual_gates import VirtualMove, WireCut, VIRTUAL_GATE_TYPES
from qvm.virtual_circuit import VirtualCircuit, generate_instantiations

from HwAwareCutter.Logger import Logger


# Helper classes
@dataclass
class DagVertex:
    idx : int
    qubit : Qubit
    nThGate : int
    opNode : DAGOpNode
    opNodeV0Idx : int
    opNodeV1Idx : int
class EdgeType(Enum):
        GateCut = 0
        WireCut = 1


class Cutter:
    # maxNQpdCuts: only use QPD cuts up to this amount. Afterward, teleportation will be used for cutting.
    def __init__(self, inputCirc : QuantumCircuit, maxNPartitions : int = 2, maxNQubitsPerPartition : int | List[int] = 10, forceNWireCuts : int | None = None, forceNGateCuts : int | None = None, maxNQpdCuts : int | None = None, maxNCuts : int | None = None, maxCutsPerPartitions : int | None = None) -> None:
        self.logger = Logger().getLogger(__name__)
        self.inputCirc = inputCirc.copy()
        self.maxNPartitions = maxNPartitions

        if type(maxNQubitsPerPartition) == int:
            self.maxNQubitsPerPartition = [maxNQubitsPerPartition for _ in range(maxNPartitions)]
        elif type(maxNQubitsPerPartition) == list:
            self.maxNQubitsPerPartition = maxNQubitsPerPartition
        else:
            raise RuntimeError("Invalid type")

        assert(len(self.maxNQubitsPerPartition) == self.maxNPartitions)
        assert(len(inputCirc.qubits) <= sum(self.maxNQubitsPerPartition))

        self.forceNWireCuts = None
        if forceNWireCuts is not None:
            assert(forceNWireCuts>=0)
            self.forceNWireCuts = forceNWireCuts
        
        self.forceNGateCuts = None
        if forceNGateCuts is not None:
            assert(forceNGateCuts>=0)
            self.forceNGateCuts = forceNGateCuts
        
        self.maxNCuts = None
        if maxNCuts is not None:
            nWireCuts = 0 if forceNWireCuts is None else forceNWireCuts
            nGateCuts = 0 if forceNGateCuts is None else forceNGateCuts
            assert(maxNCuts>0)
            assert(maxNCuts >= nWireCuts+nGateCuts)
            self.maxNCuts = maxNCuts

        self.maxNQpdCuts = None
        if maxNQpdCuts is not None:
            assert(maxNQpdCuts>=0)
            if self.maxNCuts is not None:
                assert(maxNQpdCuts<=self.maxNCuts)
            self.maxNQpdCuts = maxNQpdCuts
        
        self.maxCutsPerPartitions = maxCutsPerPartitions
        if self.maxCutsPerPartitions is not None:
            assert(self.maxCutsPerPartitions>0)

        self.decomposedCirc = inputCirc.decompose()
        self.V, self.W, self.G, self.I = self._readCirc(self.decomposedCirc)

        self.models = None # result of the optimizer will be stored here.

        self.c = Control()
        self.program = ""
        self._addVariables()
        self._addConstraintsAndObjectives()

    
    # this function will solve the problem and return True if solution(s) exists. 
    # Return False otherwise.
    # If there're multiple solutions, each call to this function
    # will assign a different model to the internal variable.
    # After calling `solve()` and a True is returned, call `getCutCirc()` to get cut circuit
    # and/or call `getModelKeyResuts()` to get solution key results.
    # NOTE: solve() might return duplicated solutions.
    def solve(self) -> bool:
        self.c.configuration.solve.models = 0
        self.c.add("base", [], self.program)
        self.c.ground([("base", [])])
        
        solveHandle = self.c.solve(yield_=True)
        self.models = None
        for model in solveHandle:  # type: ignore
            self.models = model

        if self.models is None:
            return False

        modelList = list(self.models.symbols(shown=True))

        # {pIdx:[vIdx]}
        self.o_vp = defaultdict(list)
        # eIdx
        self.c_eIdx = set()
        self.b_eIdx = set()
        self.C_p = [None for _ in range(self.maxNPartitions)]
        self.Q_p = [None for _ in range(self.maxNPartitions)]
        self.S = None
        self.Q = None
        self.C = None
        self.sumCuts = None
        self.sumWireCuts = None
        self.sumGateCuts = None
        self.sumQpdCuts = None
        self.maxQpdEdgeIdx = None
        self.minTeleEdgeIdx = None

        for m in modelList:
            if m.name == "assign":
                vIdx = m.arguments[0].number-1
                pIdx = m.arguments[1].number-1
                self.o_vp[pIdx].append(vIdx)
            if m.name == "c_e":
                eIdx = m.arguments[0].number-1
                self.c_eIdx.add(eIdx)
            if m.name == "b_e":
                eIdx = m.arguments[0].number-1
                self.b_eIdx.add(eIdx)
            if m.name == "c":
                pIdx = m.arguments[0].number-1
                c = m.arguments[1].number
                self.C_p[pIdx] = c
            if m.name == "q":
                pIdx = m.arguments[0].number-1
                q = m.arguments[1].number
                self.Q_p[pIdx] = q
            if m.name == "s":
                self.S = m.arguments[0].number
            if m.name == "bigQ":
                self.bigQ = m.arguments[0].number
            if m.name == "bigC":
                self.bigC = m.arguments[0].number
            if m.name == "sumCuts":
                self.sumCuts = m.arguments[0].number
            if m.name == "sumWireCuts":
                self.sumWireCuts = m.arguments[0].number
            if m.name == "sumGateCuts":
                self.sumGateCuts = m.arguments[0].number
            if m.name == "sumQpdCuts":
                self.sumQpdCuts = m.arguments[0].number
            if m.name == "maxQpdEdgeIdx":
                self.maxQpdEdgeIdx = m.arguments[0].number
            if m.name == "minTeleEdgeIdx":
                self.minTeleEdgeIdx = m.arguments[0].number
            
        return True
    

    # return decomposed-circuit, cut-marked-circuit, cut-marked-with-move-gates-circuit, cut-circuit, instantiated-circuits
    def getResultCircs(self, getInstantiations : bool = False) -> Tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit, List[List[QuantumCircuit]]]:
        if self.models is None:
            raise RuntimeError("no model exists")

        copiedDecomposedCirc = self.decomposedCirc.copy()
        # update V
        V, _, _, _ = self._readCirc(copiedDecomposedCirc)
        beforeCutDag = circuit_to_dag(copiedDecomposedCirc, False)

        
        markedDag = self._repaceGateCutsAndMarkWireCuts(beforeCutDag, V)
        markedCircToShow = dag_to_circuit(markedDag)
        markedQvmDag = DAG(dag_to_circuit(markedDag, False), False)

        vmoveToVIdxMapping, moveQubits = self._replaceWireCutMarkWithVirtualMoveGates(markedQvmDag)
        markedCircWithVirtualMoves = markedQvmDag.to_circuit()
        markedCircWithVirtualMovesToShow = markedCircWithVirtualMoves.copy()

        fragments = self._getFragments(V, set(markedCircWithVirtualMoves.qubits), vmoveToVIdxMapping, moveQubits)

        self.logger.debug("fragments:")
        for idx, frag in enumerate(fragments):
            qubitNames = [f"{q.register.name}{q.index}" for q in frag]
            self.logger.debug(f"    {idx}: {qubitNames}")

        qubitMapping = markedQvmDag.fragment(fragments)
        self.logger.debug("qubit mapping:")
        for old, new in qubitMapping.items():
            self.logger.debug(f"    {old.register.name}{old.index} => {new.register.name}_{new.index}")

        cutCirc = markedQvmDag.to_circuit()

        return self.decomposedCirc, markedCircToShow, markedCircWithVirtualMovesToShow, cutCirc, [] if not getInstantiations else self._generateInstantiation(VirtualCircuit(cutCirc.copy()))
    

    # return S, A, L, nWireCuts, nGateCuts, Q, [Q_p1, Q_p2, ..., Q_pn], C, [C_p1, C_p2, ..., C_pn]
    def getModelKeyResults(self) -> Tuple[int, int, int, int, List[int]]:
        if self.models is None:
            raise RuntimeError("no model exists")
        
        return self.S, None, None, self.sumWireCuts, self.sumGateCuts, self.Q, self.Q_p, self.C, self.C_p

    
    def logOptimizerResults(self) -> None:
        self.logger.debug("O_vp results: ")
        for p, v in self.o_vp.items():
            self.logger.debug(f"    {p}: {v}")

        self.logger.debug(f"b_eIdx:    {self.b_eIdx}")

    
    # read circuit and return V, W, G, I according to the paper
    def _readCirc(self, circuit : QuantumCircuit) -> Tuple[List[DagVertex], List, List, List[DagVertex]]:
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
            # skip barriers and non-2qubit-gates
            if len(v.qargs) != 2 or v.op.name == "barrier" or isinstance(v.op, VirtualMove):
                continue
            qubit0 = v.qargs[0]
            qubit1 = v.qargs[1]
            assert(len(circuit.find_bit(qubit0).registers)==1)
            assert(len(circuit.find_bit(qubit1).registers)==1)
            v0Idx = len(V)
            v1Idx = v0Idx + 1
            if v.op.label is None:
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

        self._checkGraph(V, W, G, I)
        return V, W, G, I


    # check if graph is valid
    def _checkGraph(self, V, W, G, I):
        edges = W + G
        nVertices = len(V)
        verticesSet = set()

        for (u, v) in edges:
            assert(u < v)
            assert(u < nVertices)
            verticesSet.add(u)
            verticesSet.add(v)
        
        assert(verticesSet == set(range(nVertices)))
        assert(set([v.idx for v in I]).issubset(set(v.idx for v in V)))

    
    def _addVariables(self):
        # circuit is cut at edge e
        self.c_e = []

        self._populateVariables()

    def _populateVariables(self):
        for v in self.I:
            self.program += self._getTerm(f"i({v.idx+1})")
        for vIdx in range(len(self.V)):
            self.program += self._getTerm(f"v({vIdx+1})")
        for pIdx in range(self.maxNPartitions):
            self.program += self._getTerm(f"p({pIdx+1})")
        for idx, n in enumerate(self.maxNQubitsPerPartition):
            self.program += self._getTerm(f"maxNQubitsPerPartition({idx+1}, {n})")
        if self.maxNQpdCuts is not None:
            self.program += self._getTerm(f"maxNQpdCuts({self.maxNQpdCuts})")
        if self.maxNCuts is not None:
            self.program += self._getTerm(f"maxNCuts({self.maxNCuts})")
        self.program += self._getTerm(f"maxNPartitions({self.maxNPartitions})")
        if self.maxCutsPerPartitions is not None:
            self.program += self._getTerm(f"maxCutsPerPartitions({self.maxCutsPerPartitions})")
        self.program += self._getTerm(f"nVertices({len(self.V)})")
            
        # populate c_e, b_e
        for eIdx in range(len(self.W)):
            c_eIdx = len(self.c_e)
            u, v = self.W[eIdx]
            self.c_e.append({
                "eIdx" : eIdx,
                "edge" : self.W[eIdx],
                "edgeType" : EdgeType.WireCut
            })
            self.program += self._getTerm(f"edge({c_eIdx+1}, {u+1}, {v+1})")
            self.program += self._getTerm(f"w({c_eIdx+1})")
        for eIdx in range(len(self.G)):
            u, v = self.G[eIdx]
            vertex = self.V[u]
            # This is not part of the original paper.
            # If a gate is not supported for Virtualization => Don't cut it.
            if vertex.opNode.op.name not in VIRTUAL_GATE_TYPES:
                continue
            c_eIdx = len(self.c_e)
            self.c_e.append({
                "eIdx" : eIdx,
                "edge" : self.G[eIdx],
                "edgeType" : EdgeType.GateCut
            })
            self.program += self._getTerm(f"edge({c_eIdx+1}, {u+1}, {v+1})")
            self.program += self._getTerm(f"g({c_eIdx+1})")
        assert(self.maxNPartitions <= len(self.V))


    def _addConstraintsAndObjectives(self):

        # c_e constraints
        self.program += \
            self._getTerm(f"c_e(IDX) :- edge(IDX, U, V), assign(U,P1), assign(V,P2), P1!=P2")
        
        # b_e implies c_e
        self.program += self._getTerm("c_e(A) :- b_e(A)")
        
        #o_vp 1st constraints: no vertex is assigned twice => don't need. Check later.
        # o_vp 2nd constraints: at least 1 partition is assigned to each vertex v
        self.program += self._getTerm(f"{{assign(V, P) : v(V), p(P)}} = 1 :- nVertices(N), V=1..N")
    
        # Q_p constraints        
        firstTerm = f"1,1,VIDX : i(VIDX), assign(VIDX, P)"
        secondTerm = f"1,2,EIDX : w(EIDX), edge(EIDX, U, V), assign(V, P)"
        thirdTerm = f"1,3,EIDX : b_e(EIDX), edge(EIDX, U, V), assign(U, P); 1,4 : b_e(EIDX), edge(EIDX, U, V), not assign(U, P), assign(V, P)"
        s = f"q(P, S) :- S = #sum{{ {firstTerm}; {secondTerm}; {thirdTerm} }}, assign(_, P), maxNQubitsPerPartition(P, N), maxNPartitions(NP), S <= N, P=1..NP"
        self.program += self._getTerm(s)

        # C_p constraints
        firstTerm = f"1,1,EIDX : c_e(EIDX), not b_e(EIDX), edge(EIDX, U, V), assign(U, P)"
        secondTerm = f"1,2,EIDX : c_e(EIDX), not b_e(EIDX), edge(EIDX, U, V), not assign(U, P), assign(V, P)"
        s = f"c(P, S) :- S = #sum{{ {firstTerm}; {secondTerm} }}, maxNPartitions(M), P=1..M"
        self.program += self._getTerm(s)
        # TODO: FIXME
        # if self.maxCutsPerPartitions is not None:
        #     self.program += self._getTerm(f":- c(_, S), maxCutsPerPartitions(M), S <= M")

        # NOTE: only calculate sampling overhead for now.
        # clingo ASP currently not yet support aggregates for product => use sum
        GATE_CUT_QPD_COST = 6
        WIRE_CUT_QPD_COST = 8
        TELE_COST = 0 # 1

        # TODO: check
        firstTerm = f"{GATE_CUT_QPD_COST}, 1, EIDX : not b_e(EIDX), c_e(EIDX), g(EIDX)"
        secondTerm = f"{WIRE_CUT_QPD_COST}, 2, EIDX : not b_e(EIDX), c_e(EIDX), w(EIDX)"
        thirdTerm = f"{TELE_COST}, 3, EIDX : b_e(EIDX)"
        self.program += self._getTerm(f"s(S) :- S = #sum{{ {firstTerm}; {secondTerm}; {thirdTerm}}}")

        # helper constraints : force N wire cuts.
        if self.forceNWireCuts is not None or self.maxNCuts is not None:
            s = f"sumWireCuts(S) :- S = #sum{{ 1, EIDX : c_e(EIDX), w(EIDX) }}"
            if self.forceNWireCuts is not None:
                s += f", S == {self.forceNWireCuts}"
            self.program += self._getTerm(s)
        # helper constraints : force N gate cuts.
        if self.forceNGateCuts is not None or self.maxNCuts is not None:
            s = f"sumGateCuts(S) :- S = #sum{{ 1, EIDX: c_e(EIDX), g(EIDX) }}"
            if self.forceNGateCuts is not None:
                s += f", S == {self.forceNGateCuts}"
            self.program += self._getTerm(s)
        
        self.program += self._getTerm(f"sumCuts(S) :- S = #sum{{ G, 1: sumGateCuts(G); W, 2 : sumWireCuts(W)  }}")
        # TODO: FIXME
        # if self.maxNCuts is not None:
        #     self.program += self._getTerm(f":- sumCuts(S), S <= {self.maxNCuts}")

        self.program += self._getTerm(f"sumQpdCuts(S) :- S = #sum{{ 1, 1, EIDX: c_e(EIDX), not b_e(EIDX) }}")
        # TODO: FIXME
        # if self.maxNQpdCuts is not None:            
        #     self.program += self._getTerm(f":- sumQpdCuts(S), S <= {self.maxNQpdCuts}")
        

        # TODO: check
        self.program += self._getTerm(f"b_e(EIDX) :- c_e(EIDX), sumQpdCuts(S1), maxNQpdCuts(S2), S1==S2")
        
        self.program += self._getTerm(f"maxQpdEdgeIdx(X) :- X = #max{{ EIDX : c_e(EIDX), not b_e(EIDX); 0 : not c_e(_) }} ")
        self.program += self._getTerm(f"minTeleEdgeIdx(X) :- X = #min{{ EIDX : b_e(EIDX); {len(self.V)+1} : not b_e(_) }}")
        # TODO: check
        # self.program += self._getTerm(f":- maxQpdEdgeIdx(A), minTeleEdgeIdx(B), A < B")
        
        self.program += self._getTerm("bigQ(Q) :- Q=#max{N : q(_, N)}")
        self.program += self._getTerm("bigC(C) :- C=#max{N : c(_, N)}")
        # self.program += self._getTerm("#minimize {S, Q, C : s(S), bigQ(Q), bigC(C)}")
        # self.program += self._getTerm("#minimize {S: s(S)}")

        self.program += self._getTerm("#show assign/2")
        self.program += self._getTerm("#show c_e/1")
        self.program += self._getTerm("#show b_e/1")
        self.program += self._getTerm("#show c/2")
        self.program += self._getTerm("#show q/2")
        self.program += self._getTerm("#show s/1")
        self.program += self._getTerm("#show bigQ/1")
        self.program += self._getTerm("#show bigC/1")
        self.program += self._getTerm("#show sumCuts/1")
        self.program += self._getTerm("#show sumWireCuts/1")
        self.program += self._getTerm("#show sumGateCuts/1")
        self.program += self._getTerm("#show sumQpdCuts/1")
        self.program += self._getTerm("#show maxQpdEdgeIdx/1")
        self.program += self._getTerm("#show minTeleEdgeIdx/1")

        self.logger.debug(f"ASP program ==========================================")
        self.logger.debug(self.program)
        self.logger.debug(f"ASP program ========================================== END")
    

    # FIXME: teleport is not yet supported. Currently VirtualGate and MoveGate are used.
    def _repaceGateCutsAndMarkWireCuts(self, dag : DAGCircuit, V : List[DagVertex]) -> DAGCircuit:
        for eIdx in self.c_eIdx:
            c_eVar = self.c_e[eIdx]

            uIdx, vIdx = c_eVar["edge"]
            u = V[uIdx]
            v = V[vIdx]
            if c_eVar["edgeType"] == EdgeType.GateCut:
                gateName = f"{v.opNode.name} {v.opNode.op.label}"
                if eIdx in self.b_eIdx:
                    gateName += " TELE"
                dag.substitute_node(u.opNode, VIRTUAL_GATE_TYPES[v.opNode.name](u.opNode.op, gateName))
                self.logger.info(f"GateCut {gateName} is replaced.")
            elif c_eVar["edgeType"] == EdgeType.WireCut:
                newDag = DAGCircuit()
                newDag.add_qubits(u.opNode.qargs)
                newDag.apply_operation_back(op=u.opNode.op, qargs=u.opNode.qargs)
                wireCutLabel = f"{uIdx}_{vIdx}"
                if eIdx in self.b_eIdx:
                    wireCutLabel += " TELE"
                wirecutOp = WireCut(1, f"WC {wireCutLabel}")
                wirecutOp.wireCutLabel = wireCutLabel
                newDag.apply_operation_back(op=wirecutOp, qargs=[u.qubit])
                self.logger.info(f"WireCut {wireCutLabel} is marked.")
                newNodesMap = dag.substitute_node_with_dag(u.opNode, newDag)
                newNode = [*newNodesMap.values()][0]
                u0 = V[u.opNodeV0Idx]
                u1 = V[u.opNodeV1Idx]
                assert(u.opNode == u0.opNode == u1.opNode)
                assert(newNode.op.name == u.opNode.op.name == u0.opNode.op.name == u1.opNode.op.name)
                assert(newNode.op.label == u.opNode.op.label == u0.opNode.op.label == u1.opNode.op.label)
                assert(newNode.qargs == u.opNode.qargs == u0.opNode.qargs == u1.opNode.qargs)
                u0.opNode = newNode
                u1.opNode = newNode
        return dag
    

    def _replaceWireCutMarkWithVirtualMoveGates(self, dag: DAG) -> Tuple[List[int], List[Qubit]]:

        moveQubits = []
        vmoveToVIdxMapping = []

        if self.sumWireCuts == 0:
            return vmoveToVIdxMapping, moveQubits

        move_reg = QuantumRegister(self.sumWireCuts, "vmove")
        dag.add_qreg(move_reg)
        qubit_mapping: dict[Qubit, Qubit] = {}
        cut_ctr = 0
        def _find_qubit(qubit: Qubit) -> Qubit:
            while qubit in qubit_mapping:
                qubit = qubit_mapping[qubit]
            return qubit
        for node in nx.topological_sort(dag):
            instr = dag.get_node_instr(node)
            instr.qubits = [_find_qubit(qubit) for qubit in instr.qubits]

            if isinstance(instr.operation, WireCut):
                instr.operation = VirtualMove(SwapGate(label=instr.operation.wireCutLabel))
                instr.qubits.append(move_reg[cut_ctr])
                qubit_mapping[instr.qubits[0]] = instr.qubits[1]
                # get the vIdx on the right hand side of WC mark.
                vIdxStr = instr.operation.label.split()[-1].split("_")[-1]
                vIdx = int(vIdxStr)
                vmoveToVIdxMapping.append(vIdx)
                moveQubits.append(instr.qubits[1])
                cut_ctr += 1
        
        return vmoveToVIdxMapping, moveQubits
    

    def _getFragments(self, V : List[DagVertex], qubits : Set[Qubit], vmoveToVIdxMapping : List[int], moveQubits : List[Qubit]) -> List[Set[Qubit]]:
        results = [set() for _ in range(self.maxNPartitions)]
        visited = set()

        vIdxToPIdxMapping = {}

        for pIdx, vIdxs in self.o_vp.items():

            for vIdx in vIdxs:
                vIdxToPIdxMapping[vIdx] = pIdx
                v = V[vIdx]
                q = v.qubit

                if q in visited or q in moveQubits:
                    continue

                visited.add(q)
                results[pIdx].add(q)
        
        # fragment move qubits
        for vMoveIdx, vIdx in enumerate(vmoveToVIdxMapping):
            moveQubit = moveQubits[vMoveIdx]
            pIdx = vIdxToPIdxMapping[vIdx]
            results[pIdx].add(moveQubit)
            visited.add(moveQubit)
        
        # there exists qubits without any gates => no edges from our algorithm.
        leftOverQubits = qubits - visited
        availableSpots = 0

        for idx in range(self.maxNPartitions):
            availableSpots += self.maxNQubitsPerPartition[idx] - len(results[idx])

        if availableSpots < len(leftOverQubits):
            raise RuntimeError("not enough available spots")
        
        for idx in range(self.maxNPartitions):
            availableSpotsInThisPartition = self.maxNQubitsPerPartition[idx] - len(results[idx])
            if availableSpotsInThisPartition == 0:
                continue

            qubitsToAdd = []
            while availableSpotsInThisPartition>0 and len(leftOverQubits)>0:
                qubitsToAdd.append(leftOverQubits.pop())
                availableSpotsInThisPartition-=1

            if qubitsToAdd:
                self.logger.info(f"added {len(qubitsToAdd)} left over qubits {qubitsToAdd} to partition {idx}")
                results[idx].update(qubitsToAdd)

        return results
    

    def _generateInstantiation(self, virt: VirtualCircuit) -> List[List[QuantumCircuit]]:
        instantiations = []

        for frag, frag_circuit in virt.fragment_circuits.items():
            instance_labels = virt.get_instance_labels(frag)
            instantiations.append(generate_instantiations(frag_circuit, instance_labels))
    

    def _getTerm(self, term : str) -> str:
        return f"{term}.\n"