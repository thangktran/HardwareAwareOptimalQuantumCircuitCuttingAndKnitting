import logging
from time import perf_counter
from dataclasses import dataclass
from multiprocessing import Pool

from qiskit.providers import Job
from qiskit.circuit import QuantumRegister as Fragment
from qiskit.providers.fake_provider import *
from qiskit import transpile

from qvm.virtual_circuit import VirtualCircuit, generate_instantiations
from qvm.quasi_distr import QuasiDistr

from HwAwareCutter.Logger import Logger


@dataclass
class RunTimeInfo:
    run_time: float
    knit_time: float


def run_virtual_circuit(
    virt: VirtualCircuit, shots: int = 20000
) -> tuple[dict[int, float], RunTimeInfo]:
    jobs: dict[Fragment, Job] = {}

    Logger().getLogger(__name__).info(
        f"Running virtualizer with {len(virt.fragment_circuits)} "
        + f"{tuple(circ.num_qubits for circ in virt.fragment_circuits.values())} "
        + f"fragments and {len(virt._vgate_instrs)} vgates..."
    )

    num_instances = 0
    now = perf_counter()
    for frag, frag_circuit in virt.fragment_circuits.items():
        instance_labels = virt.get_instance_labels(frag)
        instantiations = generate_instantiations(frag_circuit, instance_labels)
        num_instances += len(instantiations)
        #backend = FakePerth()
        #comp_instantiations = transpile(instantiations, optimization_level=3)
        jobs[frag] = virt.get_backend(frag).run(instantiations, shots=shots)
        #jobs[frag] = backend.run(comp_instantiations, shots=shots)

    Logger().getLogger(__name__).info(f"Running {num_instances} instances...")
    results = {}
    for frag, job in jobs.items():
        result = job.result()
        try:
            # there's case where 1 partition has 1 qubit and contains virtual gate and
            # virtual move with no measurement => get_counts() through an exception.
            # happens to benchmark with aqft of 10 qubits with exactly 1 WireCut and
            # exactly 1 GateCut
            counts = result.get_counts()
            counts = [counts] if isinstance(counts, dict) else counts
            results[frag] = [QuasiDistr.from_counts(c) for c in counts]
        except Exception:
            pass

    run_time = perf_counter() - now

    Logger().getLogger(__name__).info(f"Knitting...")

    with Pool(processes=8) as pool:
        now = perf_counter()
        res_dist = virt.knit(results, pool)
        knit_time = perf_counter() - now

    Logger().getLogger(__name__).info(f"Knitted in {knit_time:.2f}s.")

    return res_dist.nearest_probability_distribution(), RunTimeInfo(run_time, knit_time)
