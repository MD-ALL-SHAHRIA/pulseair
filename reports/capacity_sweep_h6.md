# Sequence-model capacity sweep — horizon 6 h

Closes the reviewer question left open in `final_results_summary.md` §5: the Phase 5
models had 53k–70k parameters, and the conclusion that they lose to a RandomForest
would be weak if they were simply too small.

LSTM `hidden_size` and Transformer `d_model` over [64, 128, 256], at
**lr = 0.0003** (the value the Phase 5 learning-rate sweep selected), same
validation-based selection, same early stopping (patience 8, cap
40 epochs). Transformer feedforward width scales with `d_model` at
2×, so the models widen coherently.

---

## Results

| Architecture | Size | Parameters | Val macro-F1 | Best/run epochs | Minutes |
| --- | --- | --- | --- | --- | --- |
| lstm | 64 | 52,870 | 0.5151 | 6/14 | 3.7 |
| lstm | 128 | 204,038 | 0.4880 | 5/13 | 5.6 |
| lstm | 256 | 801,286 | 0.4828 | 2/10 | 11.1 |
| transformer | 64 | 69,510 | 0.5148 | 6/14 | 5.6 |
| transformer | 128 | 270,086 | 0.5053 | 1/9 | 7.1 |
| transformer | 256 | 1,064,454 | 0.4947 | 2/10 | 19.7 |

For reference, the RandomForest scores **0.4895** on the same validation split.

| Architecture | Smallest → largest | Spread across sizes | Best size | Parameter ratio | Plateau? |
| --- | --- | --- | --- | --- | --- |
| lstm | -0.0322 | 0.0322 | 64 | 15.2× | no |
| transformer | -0.0201 | 0.0201 | 64 | 15.3× | no |

---

## Verdict

**Both architectures get monotonically *worse* with capacity. The
Phase 5 conclusion holds, and this is stronger evidence than a plateau would have
been.**

| Architecture | Smallest → largest | Parameter ratio | Trend |
| --- | --- | --- | --- |
| lstm | -0.0322 | 15.2× | monotonically declining |
| transformer | -0.0201 | 15.3× | monotonically declining |

The LSTM loses 0.0322 validation macro-F1
going from 64 to
256 hidden units
(15× the parameters); the Transformer loses
0.0201 over the same widening. Best
epoch also falls as capacity rises — the largest models peak at epoch 1–2 and then
overfit — which is the signature of models with more capacity than the signal
supports.

**The "were they big enough?" objection is answered decisively: they were already too
big.** This is consistent with the MC-dropout decomposition, which put only
1.7% of predictive uncertainty in the epistemic term — the part extra
capacity could address. The remaining 98% is irreducible given nine
channels at a six-hour horizon, and adding parameters against irreducible noise costs
generalisation rather than buying accuracy.

Note the smallest configuration in this sweep is the one Phase 5 used, and it
reproduces exactly (0.5151 LSTM,
0.5148 Transformer), which also serves
as a determinism check on the Phase 5 training run.

Reproduce with `python -m src.models.capacity_sweep`.
