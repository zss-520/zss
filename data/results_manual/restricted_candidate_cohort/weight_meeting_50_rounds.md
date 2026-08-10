# 50-round metric-weight meeting trace

Each round bootstraps datasets and records three local expert roles: statistics, screening, and review.
No model-specific priority bonus is used.
Model eligibility is decided before scoring by a model-agnostic measured-resource budget gate.
Resource-excluded models: none.
Eligible models lacking resource measurements: AMPsorter, C_AMPs-predict, HMD-AMP.

## Round 1

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, AMPsorter, HMD-AMP

## Round 2

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Recall in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, AMPsorter, C_AMPs-predict

## Round 3

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Recall in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, AMPsorter, C_AMPs-predict

## Round 4

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 5

- Sampled datasets: C_AMPs-predict_test, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 6

- Sampled datasets: Veltri_test, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports Recall in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 7

- Sampled datasets: C_AMPs-predict_test, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 8

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 9

- Sampled datasets: Veltri_test, C_AMPs-predict_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 10

- Sampled datasets: Veltri_test, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: BalancedAccuracy; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 11

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 12

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 13

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 14

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 15

- Sampled datasets: C_AMPs-predict_test, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 16

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict

## Round 17

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict

## Round 18

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict

## Round 19

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 20

- Sampled datasets: Veltri_test, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 21

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 22

- Sampled datasets: C_AMPs-predict_test, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 23

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict

## Round 24

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 25

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 26

- Sampled datasets: Veltri_test, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 27

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 28

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 29

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 30

- Sampled datasets: C_AMPs-predict_test, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Recall in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 31

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Recall in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, AMPsorter, C_AMPs-predict

## Round 32

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Recall in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, AMPsorter, C_AMPs-predict

## Round 33

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 34

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUPRC; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 35

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 36

- Sampled datasets: Veltri_test, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 37

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict

## Round 38

- Sampled datasets: C_AMPs-predict_test, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 39

- Sampled datasets: C_AMPs-predict_test, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUPRC; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 40

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUPRC; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, HMD-AMP, AMPsorter

## Round 41

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict

## Round 42

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Recall in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, AMPsorter, C_AMPs-predict

## Round 43

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict

## Round 44

- Sampled datasets: C_AMPs-predict_test, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Recall in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 45

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict

## Round 46

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict

## Round 47

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 48

- Sampled datasets: Veltri_test, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, C_AMPs-predict, AMPsorter

## Round 49

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Recall in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Precision; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, AMPsorter, C_AMPs-predict

## Round 50

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports ACC in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ACC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, HMD-AMP, C_AMPs-predict
