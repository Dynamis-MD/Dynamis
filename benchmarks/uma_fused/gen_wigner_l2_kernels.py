"""Print the explicit-term Triton kernels of dynamis_model/.../uma_fused/wigner_l2.py (_wig_l2_fwd, _wig_l2_bwd1).
Usage: python benchmarks/uma_fused/gen_wigner_l2_kernels.py > /tmp/k.py, then paste over the two kernels."""
import itertools
from collections import Counter

combos = list(itertools.combinations_with_replacement(range(4), 4))   # 35, same order as wigner_l2._COMBOS
Q = ["q0", "q1", "q2", "q3"]


def mono(cb):
    return "*".join(Q[i] for i in cb)


f = ["@triton.jit", "def _wig_l2_fwd(q_ptr, c_ptr, d_ptr, E, BLOCK: tl.constexpr):",
     "    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)", "    m = e < E", "    e64 = e.to(tl.int64)"]
f += [f"    q{v} = tl.load(q_ptr + e64 * 4 + {v}, mask=m, other=0.0)" for v in range(4)]
f += ["    j = tl.arange(0, 32)", "    acc = tl.zeros([BLOCK, 32], dtype=tl.float32)"]
f += [f"    acc += ({mono(cb)})[:, None] * tl.load(c_ptr + {k * 32} + j)[None, :]" for k, cb in enumerate(combos)]
f += ["    tl.store(d_ptr + e64[:, None] * 25 + j[None, :], acc, mask=m[:, None] & (j[None, :] < 25))"]
b = ["@triton.jit", "def _wig_l2_bwd1(q_ptr, c_ptr, dd_ptr, dq_ptr, E, BLOCK: tl.constexpr):",
     '    """1-D variant: one edge per thread, dD row in 25 registers, C35 scalars (uniform loads)"""',
     "    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)", "    m = e < E", "    e64 = e.to(tl.int64)"]
b += [f"    q{v} = tl.load(q_ptr + e64 * 4 + {v}, mask=m, other=0.0)" for v in range(4)]
b += [f"    d{j} = tl.load(dd_ptr + e64 * 25 + {j}, mask=m, other=0.0)" for j in range(25)]
b += [f"    dq{v} = tl.zeros([BLOCK], dtype=tl.float32)" for v in range(4)]
for k, cb in enumerate(combos):
    b.append("    g = " + " + ".join(f"d{j} * tl.load(c_ptr + {k * 32 + j})" for j in range(25)))
    for v, n in Counter(cb).items():
        rest = list(cb)
        rest.remove(v)
        b.append(f"    dq{v} += g * ({f'{float(n)} * ' if n > 1 else ''}{'*'.join(Q[i] for i in rest)})")
b += [f"    tl.store(dq_ptr + e64 * 4 + {v}, dq{v}, mask=m)" for v in range(4)]
print("\n".join(f) + "\n\n\n" + "\n".join(b))
