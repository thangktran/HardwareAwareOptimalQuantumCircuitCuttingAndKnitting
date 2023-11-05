- python == 3.10

- DagInNode -> DagOpNode => first Op; vertex with no incoming edges
- DagOpNode has qargs => define the vertices and the G wire
- DagOpNode -> DagOutNode 

- do we need to 
Symmetry-breaking constraints
        Force small-numbered vertices into small-numbered subcircuits:
            v0: in subcircuit 0
            v1: in subcircuit_0 or subcircuit_1
            v2: in subcircuit_0 or subcircuit_1 or subcircuit_2
            ...
        """
        for vertex in range(self.num_subcircuit):
            self.model.addConstr(
                gp.quicksum(
                    [
                        self.vertex_var[subcircuit][vertex]
                        for subcircuit in range(vertex + 1)
                    ]
                )
                == 1
            )


- objective with log and exp => Kind of not possible
```python
# why z3 can't handle exp & log constraints ref: https://stackoverflow.com/questions/70289335/power-and-logarithm-in-z3
# def z3_exp(x):
#     return 10 ** x
# z3_log = Function('log_10', RealSort(), RealSort())
# z3_x = Real('z3_log_x')
# s.add(ForAll([z3_x], z3_log(z3_exp(z3_x)) == z3_x))
# s.add(ForAll([z3_x], z3_exp(z3_log(z3_x)) == z3_x))
# sumTerms = []
# for idx in range(len(b_e)):
#     term = math.log(gamma_e**2 * If(And(c_e[idx], Not(b_e[idx])), 1, 0), 10) + math.log(gamma_e_k**2 * If(b_e[idx], 1, 0), 10)
#     sumTerms.append(term)
# s.add(S == 10 ** Sum(sumTerms))
```
- export P=4 && python -u test.py -p $P 2>&1 | tee "${P}_run.log"