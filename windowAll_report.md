# MAST pipeline — window=all report

18 trajectories, 740 steps. Model: mlx-community/Qwen3-8B-4bit (4bit=True).

## Headline

- **Recall of true mistake steps (flagged by Stage 1):** 72.2% of 18 (via history-dependent modes: 27.8%; via context-independent modes: 66.7%)
- Overall step flag rate: 44.5%  |  zero-candidate trajectories: 0.0%  |  Stage 2 UNSURE rate: 0.0%
- **Agent / step / joint accuracy:** 66.7% / 27.8% / 27.8%
- Naive majority baseline: agent 55.6% (always 'WebSurfer'), step 22.2% (always 8), joint 22.2% (always ('WebSurfer', 8))
- Agent-correct / step-wrong: 7 trajectories (58.3% of agent-correct predictions)
- Predicted-agent distribution: {'WebSurfer': 12, 'Orchestrator': 5, 'FileSurfer': 1}  (ground truth: {'WebSurfer': 10, 'Orchestrator': 6, 'FileSurfer': 1, 'Assistant': 1})
- parse_error rates — Summarizer 0.00%, Stage 1 0.00%, Stage 2 0.00%; parse errors on ground-truth steps: 0
- Cost: 117362 tokens/trajectory (2112519 total; Stage 1 avg input 2229 tokens/step), 1106.8s/trajectory

## Mode flag counts

|     |   count | rate   | history_dependent   |
|----:|--------:|:-------|:--------------------|
| 1.1 |     128 | 17.30% | False               |
| 1.2 |      20 | 2.70%  | False               |
| 1.3 |      17 | 2.30%  | True                |
| 1.4 |      24 | 3.24%  | True                |
| 1.5 |       3 | 0.41%  | False               |
| 2.1 |      95 | 12.84% | True                |
| 2.2 |      10 | 1.35%  | False               |
| 2.3 |      23 | 3.11%  | False               |
| 2.4 |      77 | 10.41% | False               |
| 2.5 |      85 | 11.49% | True                |
| 2.6 |       6 | 0.81%  | True                |
| 3.1 |      65 | 8.78%  | False               |
| 3.2 |      20 | 2.70%  | False               |
| 3.3 |     121 | 16.35% | False               |

## Comparison against earlier windows

|                        |            0 |            3 |           all |
|:-----------------------|-------------:|-------------:|--------------:|
| recall                 |     0.5      |     0.611111 |      0.722222 |
| recall_hist_modes      |     0.166667 |     0.277778 |      0.277778 |
| recall_ctx_indep_modes |     0.5      |     0.555556 |      0.666667 |
| step_flag_rate         |     0.281081 |     0.518919 |      0.444595 |
| agent_acc              |     0.722222 |     0.722222 |      0.666667 |
| step_acc               |     0.333333 |     0.277778 |      0.277778 |
| joint_acc              |     0.333333 |     0.277778 |      0.277778 |
| naive_agent            |     0.555556 |     0.555556 |      0.555556 |
| naive_step             |     0.222222 |     0.222222 |      0.222222 |
| naive_joint            |     0.222222 |     0.222222 |      0.222222 |
| zero_cand_rate         |     0        |     0        |      0        |
| unsure_rate            |     0        |     0        |      0        |
| parse_err_s1           |     0        |     0        |      0        |
| tokens/traj            | 63852        | 69213        | 117362        |
| stage1_in_tok/step     |   954        |  1059        |   2229        |
| latency/traj_s         |   171.5      |   192.2      |   1106.8      |

### Mode flag rate by window

|     |    w=0 |    w=3 |   w=all |
|----:|-------:|-------:|--------:|
| 1.1 | 0.1041 | 0.1568 |  0.173  |
| 1.2 | 0.0135 | 0.0068 |  0.027  |
| 1.3 | 0.0081 | 0.0189 |  0.023  |
| 1.4 | 0.0297 | 0.0351 |  0.0324 |
| 1.5 | 0.0014 | 0.0014 |  0.0041 |
| 2.1 | 0.0622 | 0.2568 |  0.1284 |
| 2.2 | 0.0095 | 0.0162 |  0.0135 |
| 2.3 | 0.0135 | 0.0257 |  0.0311 |
| 2.4 | 0.1162 | 0.1014 |  0.1041 |
| 2.5 | 0.0486 | 0.1554 |  0.1149 |
| 2.6 | 0.0054 | 0.0054 |  0.0081 |
| 3.1 | 0.0473 | 0.0459 |  0.0878 |
| 3.2 | 0.0135 | 0.0203 |  0.027  |
| 3.3 | 0.0851 | 0.1095 |  0.1635 |

## Skepticism checklist

- Compare each accuracy to the naive baseline above (a number near the baseline is not a result).
- Agent-correct/step-wrong share: a high share means agent accuracy is partly hollow.
- Predicted-agent distribution vs ground truth: one agent dominating = prediction-space collapse.
- Context-independent modes should be roughly flat across windows; large swings deserve investigation.