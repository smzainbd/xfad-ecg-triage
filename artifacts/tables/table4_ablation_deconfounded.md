| Evaluation Arm | DS2 Alarm Events (False Alarms) | Stage A+B FA Reduction % (95% CI) | Stage A+B Sens. Loss % | SQI-Only FA Reduction % |
|---|---|---|---|---|
| **Baseline KD Student (unmodified)** | 2850 (1754) | 30.7% (17.0–49.8) | 13.59% | 33.2% |
| **+ Saliency KD (unadjusted argmax)** | 11136 (7584) | 8.3% (2.8–19.2) | 3.83% | 38.9% |
| **Protocol A: Fixed-Beat-Pool Saliency Swap (identical 2,850 baseline alarms)** | 2850 (1754) | 3.9% (0.4–11.5) | 1.64% | 33.2% |
| **Protocol B: Validation-Rate-Matched (n_val=815 threshold transferred to DS2)** | 3247 (1174) | 3.9% (0.0–6.3) | 0.05% | 2.8% |
| **Protocol C: Exact DS2 Volume-Matched (n_test=2,850 alarms)** | 2850 (1004) | 4.6% (0.0–7.3) | 0.05% | 2.9% |