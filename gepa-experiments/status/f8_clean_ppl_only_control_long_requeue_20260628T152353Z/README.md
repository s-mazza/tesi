# F8 clean PPL-only control requeue snapshot

Created from the interrupted run on faretra at 2026-06-28T15:23:53Z.

Original run root:
`gepa-experiments/results/locked_gpu_followup_20260627Tfixed_followup_132916Z`

Interrupted job:
`F8_clean_ppl_only_control_long`

Last observed progress before stop:
`GEPA Optimization: 678/12600`, around 5%.

Important caveat:
GEPA did not write a restartable optimizer checkpoint for this partial F8 run. The
requeue script restores the same experiment configuration and starts F8 cleanly
in a new output directory. The copied partial log records exactly where the
interrupted run stopped.

Files:
- `F8_clean_ppl_only_control_long.env`: generated config used by the interrupted run.
- `manifest_snapshot.tsv`: queue manifest at snapshot time.
- `split_manifest_20260628T145617Z.json`: split manifest from the interrupted run.
- `F8_clean_ppl_only_control_long_202606281455204478.log`: partial F8 log.
- `requeue_f8_only.sh`: foreground launcher for rerunning only F8 on faretra GPUs 0 and 2.

Usage on faretra:

```bash
cd /home/mazzacano/tesi
bash gepa-experiments/status/f8_clean_ppl_only_control_long_requeue_20260628T152353Z/requeue_f8_only.sh
```
