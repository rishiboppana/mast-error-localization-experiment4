# MAST pipeline — window=0 report

56 trajectories, 1334 steps. Model: mlx-community/Qwen3-8B-4bit (4bit=True).

## Headline

- **Recall of true mistake steps (flagged by Stage 1):** 71.7% of 53 (via history-dependent modes: 11.3%; via context-independent modes: 69.8%)
- Overall step flag rate: 42.1%  |  zero-candidate trajectories: 0.0%  |  Stage 2 UNSURE rate: 0.0%
- **Agent / step / joint accuracy:** 42.9% / 16.1% / 16.1%
- Naive majority baseline: agent 17.0% (always 'WebSurfer'), step 22.6% (always 1), joint 7.5% (always ('PythonDebugging_Expert', 0))
- Agent-correct / step-wrong: 15 trajectories (62.5% of agent-correct predictions)
- Predicted-agent distribution: {'WebSurfer': 8, 'Orchestrator': 9, 'Assistant': 1, 'Computer_terminal': 5, 'ArtHistory_Expert': 1, 'MusicHistorian_Expert': 1, 'PythonDebugging_Expert': 3, 'WomenInComputerScienceHistory_Expert': 1, 'Cubing_Expert': 1, 'Polish_TV_Series_Expert': 1, 'NumericalMethods_Expert': 1, 'PublicationData_Expert': 1, 'Mesopotamian_Number_Systems_Expert': 1, 'Validation_Expert': 3, 'DataExtraction_Expert': 1, 'VideoContentAnalysis_Expert': 1, 'BiblicalScholar_Expert': 1, 'WebServing_Expert': 1, 'Neurology_Expert': 1, 'Fashion_Vogue_Expert': 1, 'Marathon_Expert': 1, 'StatisticalAnalysis_Expert': 1, 'Chess_Expert': 1, 'FinancialData_Expert': 1, 'Eateries_Expert': 1, 'DataAnalysis_Expert': 1, 'HistoricalWeatherData_Expert': 1, 'WeatherData_Expert': 1, 'Karting_Expert': 1, 'Verification_Expert': 1}  (ground truth: {'WebSurfer': 9, 'Orchestrator': 6, 'BingAPI_Expert': 1, 'Geometry_Expert': 2, 'Verification_Expert': 6, 'Validation_Expert': 2, 'ArtHistory_Expert': 1, 'Lyrics_Expert': 1, 'PythonDebugging_Expert': 4, 'WomenInComputerScienceHistory_Expert': 1, 'Cubing_Expert': 1, 'Polish_TV_Series_Expert': 1, 'PublicationData_Expert': 1, 'Mesopotamian_Number_Systems_Expert': 1, 'DataExtraction_Expert': 2, 'VideoContentAnalysis_Expert': 1, 'BiblicalScholar_Expert': 1, 'VideoAnalysis_Expert': 1, 'Neurology_Expert': 1, 'DataVerification_Expert': 2, 'Marathon_Expert': 1, 'DataAnalysis_Expert': 1, 'Chess_Expert': 1, 'FinancialData_Expert': 1, 'HistoricalWeatherData_Expert': 1, 'Statistics_Expert': 1, 'Paintball_Expert': 1, 'MartialArts_Expert': 1})
- parse_error rates — Summarizer 0.00%, Stage 1 0.00%, Stage 2 0.00%; parse errors on ground-truth steps: 0
- Cost: 43010 tokens/trajectory (2408562 total; Stage 1 avg input 1065 tokens/step), 123.8s/trajectory

## Mode flag counts

|     |   count | rate   | history_dependent   |
|----:|--------:|:-------|:--------------------|
| 1.1 |     304 | 22.79% | False               |
| 1.2 |       6 | 0.45%  | False               |
| 1.3 |       4 | 0.30%  | True                |
| 1.4 |      32 | 2.40%  | True                |
| 1.5 |      31 | 2.32%  | False               |
| 2.1 |      34 | 2.55%  | True                |
| 2.2 |      28 | 2.10%  | False               |
| 2.3 |     140 | 10.49% | False               |
| 2.4 |      69 | 5.17%  | False               |
| 2.5 |      24 | 1.80%  | True                |
| 2.6 |       5 | 0.37%  | True                |
| 3.1 |     274 | 20.54% | False               |
| 3.2 |      25 | 1.87%  | False               |
| 3.3 |     191 | 14.32% | False               |

## Skepticism checklist

- Compare each accuracy to the naive baseline above (a number near the baseline is not a result).
- Agent-correct/step-wrong share: a high share means agent accuracy is partly hollow.
- Predicted-agent distribution vs ground truth: one agent dominating = prediction-space collapse.
- Context-independent modes should be roughly flat across windows; large swings deserve investigation.