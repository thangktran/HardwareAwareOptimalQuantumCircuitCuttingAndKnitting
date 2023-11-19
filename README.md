- 
```
cd $PROJECT_DIR
pdm sync
pdm add -e .
pdm add -e third_party/qvm/ --dev
```

- objective with log and exp => Kind of not possible
```python
# why z3 can't handle exp & log constraints ref: https://stackoverflow.com/questions/70289335/power-and-logarithm-in-z3
# example to implement our own exp & log
def z3_exp(x):
    return 10 ** x
z3_log = Function('log_10', RealSort(), RealSort())
z3_x = Real('z3_log_x')
s.add(ForAll([z3_x], z3_log(z3_exp(z3_x)) == z3_x))
s.add(ForAll([z3_x], z3_exp(z3_log(z3_x)) == z3_x))
sumTerms = []
for idx in range(len(b_e)):
    term = math.log(gamma_e**2 * If(And(c_e[idx], Not(b_e[idx])), 1, 0), 10) + math.log(gamma_e_k**2 * If(b_e[idx], 1, 0), 10)
    sumTerms.append(term)
s.add(S == 10 ** Sum(sumTerms))
```
