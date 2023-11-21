- 
```
cd $PROJECT_DIR
conda create --name <NAME> python=3.10
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

- Circuits result in WireCut: \<CircuitName\> (\<nQubits\>, \<nDepth\>)
  - BV (5,1)
  - QFT (5,1)
  - AQFT (5,1)
  - Random (5,4)

- 
```
#!/usr/bin/env bash
for name in syc hwe bv qft aqft add erd; do
    for n in 10 16; do
        python benchmark.py -p 2 -q 10 $name $n 1
    done
done
```

- 
```
#!/usr/bin/env bash
for name in syc hwe bv qft aqft add erd; do
    python benchmark.py -p 2 -q 10 $name 10 1 &
    a=$!
    python benchmark.py -p 2 -q 10 $name 16 1 &
    b=$!
    wait $a
    wait $b
done
```

- bench:
  - syc (16 5) (32 1)
  - hwe (16 1) (16 5)
  - bv (16 1)
  - qft (5 1) (16 1)
  - aqft (5 1) (16 1)
  - add (4 1) (6 1)

-
```
#!/usr/bin/env bash

python benchmark.py -p 2 -q 50 syc 16 5 &
a=$!
python benchmark.py -p 2 -q 50 syc 32 1 &
b=$!
wait $a
wait $b

python benchmark.py -p 2 -q 50 hwe 16 1 &
a=$!
python benchmark.py -p 2 -q 50 hwe 16 5 &
b=$!
wait $a
wait $b

python benchmark.py -p 2 -q 50 bv 16 1 &
a=$!
python benchmark.py -p 2 -q 50 bv 16 5 &
b=$!
wait $a
wait $b

python benchmark.py -p 2 -q 50 qft 5 1 &
a=$!
python benchmark.py -p 2 -q 50 qft 16 1 &
b=$!
wait $a
wait $b

python benchmark.py -p 2 -q 50 aqft 5 1 &
a=$!
python benchmark.py -p 2 -q 50 aqft 16 1 &
b=$!
wait $a
wait $b

python benchmark.py -p 2 -q 50 add 4 1 &
a=$!
python benchmark.py -p 2 -q 50 add 6 1 &
b=$!
wait $a
wait $b
```