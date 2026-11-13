# NInfer Legacy Query replay

NInfer's `legacy` strategy captures the proven last-user Query during the first
prefill, restores the exact checkpoint at that Query's beginning, and rebuilds
the causal suffix. Ordinary first-pass prefill uses resident attention-feedback
Pressure. During replay, every required window refresh instead scores the prefix
already rebuilt using the frozen first-pass Query. Replayed tokens replace their
KV and recurrent state; they do not reuse future first-pass KV. Tool continuations
without a new user Query continue to use resident-only Pressure.

Both `--kvmem-query-max-tokens` and `--kvmem-query-replay-max-tokens` default to
`0` in NInfer. Zero means no additional token-length limit: the first option uses
the full user Query for scoring, and the second permits the full causal replay.
The model context ceiling, physical KV budget, checkpoint validity, and draft
workspace requirements remain enforced. Replay streams through the existing
budget/reserve window; the complete replay need not fit in device KV at once.

A positive Query limit selects the tail of the user Query, as before. A positive
replay limit declines a replay whose complete causal suffix exceeds that limit;
it does not truncate the replay. Existing saved explicit limits remain explicit.
The console and PowerShell launchers accept zero for NInfer and report the actual
limits. Other backends retain their existing positive-limit defaults. Layerwise
does not execute Legacy replay and does not receive these controls.

This does not change the selected Query scoring formula, aggregation, graph
propagation or LRFU settings. Mean-Q cosine retains a fixed-size aggregate per
layer. Per-token PageSoftmax retains Query rows, so full-Query capture can use
substantially more host memory for long inputs. Replay also adds another forward
over its suffix: the improvement sought is recall quality, not faster ingestion.

For the previously qualified IQ3 27B comparison, the scoring settings are
`--kvmem-query-scorer meanq-cosine --kvmem-score-agg max --kvmem-graph-alpha 0
--kvmem-lrfu --kvmem-lrfu-lambda 0.5 --kvmem-lrfu-top-frac 0.9`.
Query/replay length flags may be omitted with the new defaults.

Validation on 2026-11-13 is recorded in
[the regression report](benchmarks/ninfer-query-replay-20261113.json). Four fixed
256K RULER cases retain the previously measured full-replay answer values: 3/4
exact, 15/16 values, including the known one-value miss. Mean document time is
140.55 s and logical prompt throughput is 1874.88 tokens/s, counting each input
token once despite the second forward. Question decode is 192.91 tokens/s over
the measured first-to-last-token intervals.

`tests/ninfer-streaming-replay-regression.py` additionally exercises production
PageSoftmax with 6408 Query tokens in a 2048+1024 KV window, verifies four frozen-Q
refreshes against the evaluated prefix, and checks that a subsequent tool result
uses resident-only Pressure. This test enables transfer verification and is not
a performance benchmark.
