from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import *
import math

from z3 import *
import networkx as nx

from qiskit import QuantumRegister, transpile
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
    def __init__(self, inputCirc : QuantumCircuit, maxNPartitions : int = 2, maxNQubitsPerPartition : int | List[int] = 10, forceNWireCuts : int | None = None, forceNGateCuts : int | None = None, maxNCuts : int | None = None) -> None:
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
        
        self.decomposedCirc = inputCirc.decompose()
        self.V, self.W, self.G, self.I = self._readCirc(self.decomposedCirc)

        self.model = None # result of the optimizer will be stored here.
        self.nWireCuts = 0
        self.nGateCuts = 0

        self.s = Optimize()
        self._addZ3Variables()
        self._addZ3ConstraintsAndObjectives()

    
    # this function will solve the problem and return True if solution(s) exists. 
    # Return False otherwise.
    # If there're multiple solutions, each call to this function
    # will assign a different model to the internal variable.
    # After calling `solve()` and a True is returned, call `getCutCirc()` to get cut circuit
    # and/or call `getModelKeyResuts()` to get solution key results.
    # NOTE: solve() might return duplicated solutions.
    def solve(self) -> bool:
        self.nWireCuts = 0
        self.nGateCuts = 0
        self.model = None

        modelStatus = self.s.check()
        
        if modelStatus == unsat:
            return False
        
        self.model = self.s.model()

        # count wire cuts and gate cuts
        for c_eVar in self.c_e:
            if is_false(self.model[c_eVar]):
                continue
            if c_eVar.edgeType == EdgeType.WireCut:
                self.nWireCuts += 1
            elif c_eVar.edgeType == EdgeType.GateCut:
                self.nGateCuts += 1

        return True


    # return decomposed-circuit, cut-marked-circuit, cut-marked-with-move-gates-circuit, cut-circuit, instantiated-circuits
    def getResultCircs(self, getInstantiations : bool = False) -> Tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit, List[List[QuantumCircuit]]]:
        if self.model is None:
            raise RuntimeError("no model exists")

        copiedDecomposedCirc = self.decomposedCirc.copy()
        beforeCutDag = circuit_to_dag(copiedDecomposedCirc)
        # update V
        V, _, _, _ = self._readCirc(copiedDecomposedCirc)

        
        markedDag = self._repaceGateCutsAndMarkWireCuts(beforeCutDag, V)
        markedCirc = dag_to_circuit(markedDag)
        markedQvmDag = DAG(markedCirc)

        self._replaceWireCutMarkWithVirtualMoveGates(markedQvmDag)
        markedCircWithVirtualMoves = markedQvmDag.to_circuit()

        V, _, _, _ = self._readCirc(markedCircWithVirtualMoves)
        fragments = self._getFragments(V)

        self.logger.debug("fragments:")
        for idx, frag in enumerate(fragments):
            qubitNames = [f"{q.register.name}{q.index}" for q in frag]
            self.logger.debug(f"    {idx}: {qubitNames}")

        qubitMapping = markedQvmDag.fragment(fragments)
        self.logger.debug("qubit mapping:")
        for old, new in qubitMapping.items():
            self.logger.debug(f"    {old.register.name}{old.index} => {new.register.name}_{new.index}")

        cutCirc = markedQvmDag.to_circuit()

        return copiedDecomposedCirc, markedCirc, markedCircWithVirtualMoves, cutCirc, [] if not getInstantiations else self._generateInstantiation(VirtualCircuit(cutCirc.copy()))

    
    # return S, A, L, nWireCuts, nGateCuts, Q, [Q_p1, Q_p2, ..., Q_pn]
    def getModelKeyResults(self) -> Tuple[int, int, int, int, List[int]]:
        if self.model is None:
            raise RuntimeError("no model exists")
            
        S = self.model[self.S].as_long()
        A = self.model[self.A].as_long()
        L = self.model[self.L].as_long()
        Q = self.model[self.Q].as_long()
        Q_pArr = []
        for pIdx in range(self.maxNPartitions):
            Q_pArr.append( self.model[self.Q_p[pIdx]].as_long() )
        
        return S, A, L, self.nWireCuts, self.nGateCuts, Q, Q_pArr

    
    def logOptimizerResults(self) -> None:
        self.logger.debug("O_vp results: ")

        for o_vpVar in self.o_vp:
            if is_true(self.model[o_vpVar]):
                self.logger.debug(f"    {str(o_vpVar)} = True")

        for c_e_teleportedVar in self.c_e_teleported:
            if is_true(self.model[c_e_teleportedVar]):
                self.logger.debug(f"    {str(c_e_teleportedVar)} = True")


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


    def _addZ3Variables(self):
        # gate qubit vertex v assigned to partition p
        self.o_vp = []
        # circuit is cut at edge e
        self.c_e = []
        # there're 2 cut-operation: with QPD or teleportation.
        # c_e_teleported[idx] indicate whether edge c_e[idx] was cut using teleportation.
        # Boolean value, False == QPD, True == teleportation 
        self.c_e_teleported = []
        # total number of qubits in partition p
        self.Q_p = []
        # Q is maximum number of qubit per partition
        self.Q = Int('Q')
        # S is the total cost of cutting (overhead sampling)
        self.S = Int('S')
        # A is number of required ancilla qubits
        self.A = Int('A')
        # L is total teleportation latency
        self.L = Int('L')
        # helper object to speed up look-up.
        # o_var_lookup[vIdx][pIdx] return the z3 variable.
        self.o_var_lookup = defaultdict(dict)
        self._populateZ3Variables()


    def _populateZ3Variables(self):
        # populate o_vp
        for vIdx in range(len(self.V)):
            for pIdx in range(self.maxNPartitions):
                variableName = f"o_{vIdx}_{pIdx}"
                var = Bool(variableName)
                # z3.Bool doesn't have these properties
                # we add them to make it simpler to retrieve.
                var.vIdx = vIdx
                var.pIdx = pIdx
                self.o_vp.append(var)
                self.o_var_lookup[vIdx][pIdx] = var
        # populate c_e
        for eIdx in range(len(self.W)):
            variableName = f"c_{eIdx}[W]_{self.W[eIdx][0]}_{self.W[eIdx][1]}"
            var = Bool(variableName)
            # z3.Bool doesn't have these properties
            # we add them to make it simpler to retrieve.
            var.eIdx = eIdx
            var.edge = self.W[eIdx]
            var.edgeType = EdgeType.WireCut
            self.c_e.append(var)
            self.c_e_teleported.append(Bool(f"{variableName}_teleported"))
        for eIdx in range(len(self.G)):
            u, _ = self.G[eIdx]
            vertex = self.V[u]
            # This is not part of the original paper.
            # If a gate is not supported for Virtualization => Don't cut it.
            if vertex.opNode.op.name not in VIRTUAL_GATE_TYPES:
                continue
            variableName = f"c_{eIdx}[G]_{self.G[eIdx][0]}_{self.G[eIdx][1]}"
            var = Bool(variableName)
            # z3.Bool doesn't have these properties
            # we add them to make it simpler to retrieve.
            var.eIdx = eIdx
            var.edge = self.G[eIdx]
            var.edgeType = EdgeType.GateCut
            self.c_e.append(var)
            self.c_e_teleported.append(Bool(f"{variableName}_teleported"))
        assert(self.maxNPartitions <= len(self.V))
        # populate Q_p
        for pIdx in range(self.maxNPartitions):
            variableName = f"Q_p{pIdx}"
            var = Int(variableName)
            # z3.Int doesn't have these properties
            # we add them to make it simpler to retrieve.
            var.pIdx = pIdx
            self.Q_p.append(var)


    def _addZ3ConstraintsAndObjectives(self):
        # c_e constraints
        for idx in range(len(self.c_e)):
            c_eVar = self.c_e[idx]
            u = c_eVar.edge[0]
            v = c_eVar.edge[1]
            constraints = [self.o_var_lookup[u][p]!=self.o_var_lookup[v][p] for p in range(self.maxNPartitions)]
            self.s.add(c_eVar == Or(constraints))

            self.s.add(Implies(self.c_e_teleported[idx], c_eVar))

        # o_vp 1st constraints: no vertex is assigned twice
        for vIdx in range(len(self.V)):
            constraints = [ If(i == j,
                            True,
                            Implies( self.o_var_lookup[vIdx][i], Not(self.o_var_lookup[vIdx][j]) )
                            )
                            for i in range(self.maxNPartitions) for j in range(self.maxNPartitions)]
            self.s.add(constraints)

        # o_vp 2nd constraints: at least 1 partition is assigned to each vertex v
        for vIdx, pIdxs in self.o_var_lookup.items():
            variables = []
            for pIdx, var in pIdxs.items():
                variables.append(var)
            self.s.add(Or(variables))

        # Q_p constraints
        for pIdx in range(self.maxNPartitions):
            firstSumTerm = []
            secondSumTerm = []
            thirdSumTerm = []
            # first sum term: all o_vp with v in I
            for v in self.I:
                vIdx = v.idx
                var = self.o_var_lookup[vIdx][pIdx]
                firstSumTerm.append(If(var, 1, 0))

            # second sum term
            for c_eVar in self.c_e:
                # skip all GateCut edge since we only want wire cut edge
                if c_eVar.edgeType == EdgeType.GateCut: # belong to G
                    continue
                u, v = c_eVar.edge
                o_vpVar = self.o_var_lookup[v][pIdx]
                secondSumTerm.append(If(And(c_eVar, o_vpVar), 1, 0))

            # third sum term is removed since we don't use b_e as in the original paper.
            thirdSumTerm = []
            
            self.s.add(self.Q_p[pIdx] == Sum(firstSumTerm+secondSumTerm+thirdSumTerm))
            
        GATE_CUT_QPD_COST = {
            "overheadSampling" : 6,
            "ancilla" : 0,
            "teleportLatency" : 0
        }
        WIRE_CUT_QPD_COST = {
            "overheadSampling" : 8,
            "ancilla" : 1,
            "teleportLatency" : 0
        }
        GATE_CUT_TELE_COST = {
            "overheadSampling" : 1,
            "ancilla" : 2, # TODO: check value
            "teleportLatency" : 10 # NOTE: use 10 to counter the scale different from overheadSampling.
        }
        WIRE_CUT_TELE_COST = {
            "overheadSampling" : 1,
            "ancilla" : 2, # TODO: check value
            "teleportLatency" : 10 # NOTE: use 10 to counter the scale different from overheadSampling.
        }
        
        totalOverheadSampling = 1
        totalAncilla = 0
        totalTeleportLatency = 0


        for idx in range(len(self.c_e)):
            
            if self.c_e[idx].edgeType == EdgeType.GateCut:
                
                overheadSamplingCost = If(self.c_e_teleported[idx], GATE_CUT_TELE_COST["overheadSampling"], GATE_CUT_QPD_COST["overheadSampling"])
                totalOverheadSampling *= If(self.c_e[idx], overheadSamplingCost, 1)

                ancillaCost = If(self.c_e_teleported[idx], GATE_CUT_TELE_COST["ancilla"], GATE_CUT_QPD_COST["ancilla"])
                totalAncilla += If(self.c_e[idx], ancillaCost, 0)

                latencyCost = If(self.c_e_teleported[idx], GATE_CUT_TELE_COST["teleportLatency"], GATE_CUT_QPD_COST["teleportLatency"])
                totalTeleportLatency += If(self.c_e[idx], latencyCost, 0)

            elif self.c_e[idx].edgeType == EdgeType.WireCut:
                
                overheadSamplingCost = If(self.c_e_teleported[idx], WIRE_CUT_TELE_COST["overheadSampling"], WIRE_CUT_QPD_COST["overheadSampling"])
                totalOverheadSampling *= If(self.c_e[idx], overheadSamplingCost, 1)

                ancillaCost = If(self.c_e_teleported[idx], WIRE_CUT_TELE_COST["ancilla"], WIRE_CUT_QPD_COST["ancilla"])
                totalAncilla += If(self.c_e[idx], ancillaCost, 0)

                latencyCost = If(self.c_e_teleported[idx], WIRE_CUT_TELE_COST["teleportLatency"], WIRE_CUT_QPD_COST["teleportLatency"])
                totalTeleportLatency += If(self.c_e[idx], latencyCost, 0)

            else:
                raise RuntimeError("unsupported type of cut")
            
        # NOTE: an extra variable `totalOverheadSampling` is required.
        # if use self.S directly, the program does NOT terminate.
        self.s.add(self.S == totalOverheadSampling)
        self.s.add(self.S > 1)
        self.s.add(self.A == totalAncilla * totalOverheadSampling)
        self.s.add(self.L == totalTeleportLatency)

        for pIdx in range(self.maxNPartitions):
            self.s.add(self.Q >= self.Q_p[pIdx])
            self.s.add(self.Q_p[pIdx] <= self.maxNQubitsPerPartition[pIdx])
        
        sumWireCuts = None
        sumGateCuts = None
        # helper constraints : force N wire cuts.
        if self.forceNWireCuts is not None or self.maxNCuts is not None:
            sumWireCuts = [If(c_e, 1, 0) for c_e in self.c_e if c_e.edgeType == EdgeType.WireCut]
        # helper constraints : force N gate cuts.
        if self.forceNGateCuts is not None or self.maxNCuts is not None:
            sumGateCuts = [If(c_e, 1, 0) for c_e in self.c_e if c_e.edgeType == EdgeType.GateCut]
        
        if self.forceNWireCuts is not None:
            self.s.add(Sum(sumWireCuts) == self.forceNWireCuts)
        if self.forceNGateCuts is not None:
            self.s.add(Sum(sumGateCuts) == self.forceNGateCuts)
        if self.maxNCuts is not None:
            self.s.add(Sum(sumWireCuts)+Sum(sumGateCuts) <= self.maxNCuts)

        # objectives
        self.s.minimize(self.Q)
        self.s.minimize(self.S)
        self.s.minimize(self.A)
        self.s.minimize(self.L)
        
    
    # FIXME: teleport is not yet supported. Currently VirtualGate and MoveGate are used.
    def _repaceGateCutsAndMarkWireCuts(self, dag : DAGCircuit, V : List[DagVertex]) -> DAGCircuit:
        for idx in range(len(self.c_e)):
            c_eVar = self.c_e[idx]

            if not is_true(self.model[c_eVar]):
                continue
            uIdx, vIdx = c_eVar.edge
            u = V[uIdx]
            v = V[vIdx]
            if c_eVar.edgeType == EdgeType.GateCut:
                gateName = f"{v.opNode.name} {v.opNode.op.label}"
                if is_true(self.model[self.c_e_teleported[idx]]):
                    gateName += " TELE"
                dag.substitute_node(u.opNode, VIRTUAL_GATE_TYPES[v.opNode.name](u.opNode.op, gateName))
                self.logger.info(f"GateCut {gateName} is replaced.")
            elif c_eVar.edgeType == EdgeType.WireCut:
                newDag = DAGCircuit()
                newDag.add_qubits(u.opNode.qargs)
                newDag.apply_operation_back(op=u.opNode.op, qargs=u.opNode.qargs)
                wireCutLabel = f"{uIdx}_{vIdx}"
                if is_true(self.model[self.c_e_teleported[idx]]):
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


    def _replaceWireCutMarkWithVirtualMoveGates(self, dag: DAG):
        if self.nWireCuts == 0:
            return 

        move_reg = QuantumRegister(self.nWireCuts, "vmove")
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
                cut_ctr += 1
    

    def _getFragments(self, V : List[DagVertex]) -> List[Set[Qubit]]:
        results = [set() for _ in range(self.maxNPartitions)]
        visited = set()

        for o_vpVar in self.o_vp:
            if is_false(self.model[o_vpVar]):
                continue
            pIdx = o_vpVar.pIdx
            vIdx = o_vpVar.vIdx
            v = V[vIdx]
            q = v.qubit

            if q in visited:
                continue

            visited.add(q)
            results[pIdx].add(q)

        return results
    
    def _generateInstantiation(self, virt: VirtualCircuit) -> List[List[QuantumCircuit]]:
        instantiations = []

        for frag, frag_circuit in virt.fragment_circuits.items():
            instance_labels = virt.get_instance_labels(frag)
            instantiations.append(generate_instantiations(frag_circuit, instance_labels))
        
        return instantiations