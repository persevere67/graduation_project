# Thesis Results Summary

## Core Results Table

| Experiment | Config | Final AUC | Final MRR | Best Round | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Centralized Baseline | attention, 5 epochs | 0.6044 | 0.3126 | - | Offline evaluation on saved checkpoint |
| Federated Main | attention, sigma=0, 10 rounds | 0.6411 | 0.5692 | 5 | Main federated experiment |
| Federated DP | attention, sigma=0.001, 10 rounds | 0.6441 | 0.5734 | 10 | Light noise injection |
| Federated DP | attention, sigma=0.01, 10 rounds | 0.6445 | 0.5717 | 10 | Higher noise injection |
| Federated Ablation | mean pooling, sigma=0, 10 rounds | 0.5212 | 0.4751 | 5 | Attention removed |

## Key Comparisons

### 1. Centralized vs Federated

- Centralized baseline: `AUC=0.6044`, `MRR=0.3126`
- Federated main experiment: `AUC=0.6411`, `MRR=0.5692`

Observation:

- In the current experiment setting, the federated model outperformed the centralized baseline on both AUC and MRR.
- This suggests that the proposed federated training framework can achieve competitive ranking quality while preserving the decentralized training pattern.

### 2. DP Noise Comparison

- No DP: `AUC=0.6411`, `MRR=0.5692`
- Sigma `0.001`: `AUC=0.6441`, `MRR=0.5734`
- Sigma `0.01`: `AUC=0.6445`, `MRR=0.5717`

Observation:

- After adding Gaussian noise, model performance did not show an obvious decline.
- The results under `sigma=0.001` and `sigma=0.01` remained close to the no-noise setting.
- A careful thesis claim should be: noise injection preserved performance within the current experiment scope, rather than claiming that larger noise definitively improves recommendation quality.

### 3. Ablation Study

- Attention model: `AUC=0.6411`, `MRR=0.5692`
- Mean pooling model: `AUC=0.5212`, `MRR=0.4751`

Observation:

- Removing the attention-based user encoder caused a large performance drop.
- This supports the claim that the attention mechanism is an effective component for user interest modeling in the proposed method.

## Suggested Thesis Wording

### Result Analysis Draft

In the centralized baseline experiment, the model achieved an AUC of 0.6044 and an MRR of 0.3126 on the validation set. Under the same dataset setting, the federated main experiment reached an AUC of 0.6411 and an MRR of 0.5692 after 10 communication rounds, showing better ranking performance than the baseline model. This indicates that the proposed federated learning framework can maintain strong recommendation accuracy while following a privacy-preserving decentralized training paradigm.

To further evaluate the effect of privacy enhancement, Gaussian noise was injected into the federated training process. When the noise scale was set to 0.001, the final model achieved an AUC of 0.6441 and an MRR of 0.5734. When the noise scale increased to 0.01, the final AUC was 0.6445 and the MRR was 0.5717. These results show that, within the current experimental range, the introduced noise did not cause a significant performance drop, suggesting that the proposed method has a certain degree of robustness under privacy-enhanced training.

In the ablation study, replacing the attention-based user encoder with mean pooling reduced the final AUC to 0.5212 and the MRR to 0.4751. Compared with the full federated model, the recommendation performance dropped substantially. This demonstrates that the attention mechanism plays an important role in capturing user interest from historical click behavior and is a key factor behind the effectiveness of the proposed model.

## Data Sources

- Runtime experiment summary: [runtime_experiment_summary.csv](/F:/ graduation_project/experiments/runtime_experiment_summary.csv:1)
- Runtime round metrics: [runtime_round_metrics.csv](/F:/ graduation_project/experiments/runtime_round_metrics.csv:1)
