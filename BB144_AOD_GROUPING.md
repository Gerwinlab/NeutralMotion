# How AOD grouping differs between the BB144 script and `naive_n_dag`

The biggest difference is that **the script does not discover the AOD membership automatically**. I used the known BB144 structure to choose the moving atoms, then optimized their route. Your scheduler tackles the more general problem of choosing movers from arbitrary gates.

There are two distinct kinds of grouping here:

| Group | What it means |
| --- | --- |
| **AOD load** | Atoms that stay loaded and move together |
| **Gate batch** | A subset of interactions performed at one stop |

The atoms in an AOD load do **not** need to participate in the same gate batch—or initially need to move toward partners in the same direction.

In `emit_combined` in [bb144_paper_schedule.py](bb144_paper_schedule.py), the load is explicitly:

```python
check_ids = range(144, 288)
```

All 144 check atoms move together, while all 144 data atoms stay stationary. This works because every CZ connects a check atom to a data atom: every interaction always has exactly one loaded operand. The array also fits the configured AOD span.

At any stop, only the checks whose gates are ready and whose partners are adjacent participate. The other loaded checks simply come along.

Your displacement idea appears in the next step. In `groups_and_precedence` in [bb144_paper_schedule.py](bb144_paper_schedule.py), the script computes:

```python
offset = home[data] - home[check]
```

Interactions with the **same offset** form a gate batch. One common translation brings all those pairs into interaction range. The script then finds an order for visiting those batches while preserving every qubit’s gate sequence.

Your current `_build_groups` in [dynamics.py](src/naive_n_dag/dynamics.py) makes a different decision: it groups movers from the current DAG layer using **similarity of displacement vectors**, considering either operand as the mover. For example:

- Vectors `(12, 0)` and `(16, 0)` have good alignment, but require different interaction stops.
- Vectors `(12, 0)` and `(-12, 0)` have poor alignment, but their check atoms can still share a persistent load and execute at different stops.

So alignment helps choose a locally convenient load, but it does not directly measure the cost of keeping that load through many future interactions.

Three choices give the specialized script its advantage:

1. **Consistent movers:** checks always move; data always stay trapped. This avoids future interactions where both operands end up loaded.
2. **Persistent membership:** it loads checks needed later, even if they have no currently ready gate. Under your timing model, carrying extra atoms adds no movement-time charge when the group fits.
3. **Whole-tour optimization:** it optimizes the sequence of 24 displacement groups, including interaction sides and the final return. Your reuse decision compares alternatives for the current layer with a hypothetical closing return.

That produced **6.353 ms versus 11.565 ms under identical hardware settings**. The later 2.405 ms result also benefited from changing the hardware parameters.

To bring this idea into the general scheduler, I would separate **choosing a persistent moving set** from **choosing the next gate batch**. A useful candidate is one side of a bipartite interaction graph over several upcoming layers. Then score the legal tour through its ready interaction offsets, including eventual unloading. That would let the scheduler discover opportunities like this without hard-coding the BB144 check IDs.
