# MAST pipeline — window=3 report

56 trajectories, 1334 steps. Model: mlx-community/Qwen3-8B-4bit (4bit=True).

## Headline

- **Recall of true mistake steps (flagged by Stage 1):** 88.7% of 53 (via history-dependent modes: 18.9%; via context-independent modes: 86.8%)
- Overall step flag rate: 69.9%  |  zero-candidate trajectories: 0.0%  |  Stage 2 UNSURE rate: 0.0%
- **Agent / step / joint accuracy:** 51.8% / 25.0% / 25.0%
- Naive majority baseline: agent 17.0% (always 'WebSurfer'), step 22.6% (always 1), joint 7.5% (always ('PythonDebugging_Expert', 0))
- Agent-correct / step-wrong: 15 trajectories (51.7% of agent-correct predictions)
- Predicted-agent distribution: {'WebSurfer': 12, 'Orchestrator': 3, 'BingAPI_Expert': 1, 'ProblemSolving_Expert': 1, 'Assistant': 1, 'Computer_terminal': 6, 'Lyrics_Expert': 1, 'PythonDebugging_Expert': 3, 'WomenInComputerScienceHistory_Expert': 1, 'Cubing_Expert': 1, 'Polish_TV_Series_Expert': 1, 'NumericalMethods_Expert': 1, 'PublicationData_Expert': 1, 'Mesopotamian_Number_Systems_Expert': 1, 'Geometry_Expert': 1, 'DataExtraction_Expert': 2, 'BiblicalScholar_Expert': 1, 'WebServing_Expert': 1, 'VideoAnalysis_Expert': 1, 'Literature_Expert': 1, 'DataVerification_Expert': 1, 'CelestialPhysics_Expert': 1, 'StatisticalAnalysis_Expert': 1, 'Chess_Expert': 1, 'FinancialData_Expert': 1, 'Validation_Expert': 1, 'Bioinformatics_Expert': 1, 'HistoricalWeatherData_Expert': 1, 'Verification_Expert': 1, 'WeatherData_Expert': 1, 'Karting_Expert': 1, 'NYC_Local_Expert': 1}  (ground truth: {'WebSurfer': 9, 'Orchestrator': 6, 'BingAPI_Expert': 1, 'Geometry_Expert': 2, 'Verification_Expert': 6, 'Validation_Expert': 2, 'ArtHistory_Expert': 1, 'Lyrics_Expert': 1, 'PythonDebugging_Expert': 4, 'WomenInComputerScienceHistory_Expert': 1, 'Cubing_Expert': 1, 'Polish_TV_Series_Expert': 1, 'PublicationData_Expert': 1, 'Mesopotamian_Number_Systems_Expert': 1, 'DataExtraction_Expert': 2, 'VideoContentAnalysis_Expert': 1, 'BiblicalScholar_Expert': 1, 'VideoAnalysis_Expert': 1, 'Neurology_Expert': 1, 'DataVerification_Expert': 2, 'Marathon_Expert': 1, 'DataAnalysis_Expert': 1, 'Chess_Expert': 1, 'FinancialData_Expert': 1, 'HistoricalWeatherData_Expert': 1, 'Statistics_Expert': 1, 'Paintball_Expert': 1, 'MartialArts_Expert': 1})
- parse_error rates — Summarizer 0.00%, Stage 1 0.00%, Stage 2 0.00%; parse errors on ground-truth steps: 0
- Cost: 49368 tokens/trajectory (2764621 total; Stage 1 avg input 1166 tokens/step), 148.5s/trajectory

## Mode flag counts

|     |   count | rate   | history_dependent   |
|----:|--------:|:-------|:--------------------|
| 1.1 |     326 | 24.44% | False               |
| 1.2 |       8 | 0.60%  | False               |
| 1.3 |      18 | 1.35%  | True                |
| 1.4 |      53 | 3.97%  | True                |
| 1.5 |       6 | 0.45%  | False               |
| 2.1 |     255 | 19.12% | True                |
| 2.2 |      68 | 5.10%  | False               |
| 2.3 |     136 | 10.19% | False               |
| 2.4 |      79 | 5.92%  | False               |
| 2.5 |     134 | 10.04% | True                |
| 2.6 |       6 | 0.45%  | True                |
| 3.1 |     331 | 24.81% | False               |
| 3.2 |      65 | 4.87%  | False               |
| 3.3 |     209 | 15.67% | False               |

## Comparison against earlier windows

|                        |             0 |             3 |
|:-----------------------|--------------:|--------------:|
| recall                 |     0.716981  |     0.886792  |
| recall_hist_modes      |     0.113208  |     0.188679  |
| recall_ctx_indep_modes |     0.698113  |     0.867925  |
| step_flag_rate         |     0.421289  |     0.698651  |
| agent_acc              |     0.517857  |     0.517857  |
| step_acc               |     0.25      |     0.25      |
| joint_acc              |     0.25      |     0.25      |
| naive_agent            |     0.169811  |     0.169811  |
| naive_step             |     0.226415  |     0.226415  |
| naive_joint            |     0.0754717 |     0.0754717 |
| zero_cand_rate         |     0         |     0         |
| unsure_rate            |     0         |     0         |
| parse_err_s1           |     0         |     0         |
| tokens/traj            | 45865         | 49368         |
| stage1_in_tok/step     |  1065         |  1166         |
| latency/traj_s         |   130.2       |   148.5       |

### Mode flag rate by window

|     |    w=0 |    w=3 |
|----:|-------:|-------:|
| 1.1 | 0.2279 | 0.2444 |
| 1.2 | 0.0045 | 0.006  |
| 1.3 | 0.003  | 0.0135 |
| 1.4 | 0.024  | 0.0397 |
| 1.5 | 0.0232 | 0.0045 |
| 2.1 | 0.0255 | 0.1912 |
| 2.2 | 0.021  | 0.051  |
| 2.3 | 0.1049 | 0.1019 |
| 2.4 | 0.0517 | 0.0592 |
| 2.5 | 0.018  | 0.1004 |
| 2.6 | 0.0037 | 0.0045 |
| 3.1 | 0.2054 | 0.2481 |
| 3.2 | 0.0187 | 0.0487 |
| 3.3 | 0.1432 | 0.1567 |

## Skepticism checklist

- Compare each accuracy to the naive baseline above (a number near the baseline is not a result).
- Agent-correct/step-wrong share: a high share means agent accuracy is partly hollow.
- Predicted-agent distribution vs ground truth: one agent dominating = prediction-space collapse.
- Context-independent modes should be roughly flat across windows; large swings deserve investigation.