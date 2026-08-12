# 50-round metric-weight meeting trace

Each round bootstraps datasets and records three local expert roles: statistics, screening, and review.
No model-specific priority bonus is used.
Model eligibility is decided before scoring by a model-agnostic measured-resource budget gate.
Resource-excluded models: none.
Eligible models lacking resource measurements: ai4amp, amPEPpy, ampir, amplify_bal, amplify_imb, AMPsorter, apex1.1, apin, ascan2, C_AMPs-predict, esm-AxP-GDL, HMD-AMP, iamp-ca2l, iampcn, lstm, macrel, pepnet_fast, pepnet_standard.

## Round 1

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: amplify_imb, C_AMPs-predict, AMPsorter

## Round 2

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Specificity; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, AMPsorter

## Round 3

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Specificity; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, AMPsorter

## Round 4

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 5

- Sampled datasets: C_AMPs-predict_test, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, amplify_bal, pepnet_standard

## Round 6

- Sampled datasets: Veltri_test, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, amplify_bal, ascan2

## Round 7

- Sampled datasets: C_AMPs-predict_test, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, amplify_bal, pepnet_standard

## Round 8

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Recall; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 9

- Sampled datasets: Veltri_test, C_AMPs-predict_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, amplify_bal, pepnet_standard

## Round 10

- Sampled datasets: Veltri_test, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, ascan2

## Round 11

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Recall; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 12

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Recall; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 13

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Recall; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 14

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: amplify_imb, C_AMPs-predict, AMPsorter

## Round 15

- Sampled datasets: C_AMPs-predict_test, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Recall; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 16

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, HMD-AMP

## Round 17

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, pepnet_standard

## Round 18

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, pepnet_standard

## Round 19

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 20

- Sampled datasets: Veltri_test, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 21

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, ascan2

## Round 22

- Sampled datasets: C_AMPs-predict_test, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: amplify_imb, C_AMPs-predict, AMPsorter

## Round 23

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, pepnet_standard

## Round 24

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, ascan2

## Round 25

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 26

- Sampled datasets: Veltri_test, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 27

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: amplify_imb, C_AMPs-predict, AMPsorter

## Round 28

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Recall; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 29

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Recall; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 30

- Sampled datasets: C_AMPs-predict_test, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Specificity; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, amplify_bal, ascan2

## Round 31

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, AMPsorter

## Round 32

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, AMPsorter

## Round 33

- Sampled datasets: Veltri_test, ProteoGPT_all_predictions, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, ascan2

## Round 34

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: amplify_imb, C_AMPs-predict, AMPsorter

## Round 35

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, ascan2

## Round 36

- Sampled datasets: Veltri_test, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: Specificity; missingness and metric redundancy were penalized.
- Top3: ascan2, HMD-AMP, pepnet_standard

## Round 37

- Sampled datasets: C_AMPs-predict_test, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, pepnet_standard

## Round 38

- Sampled datasets: C_AMPs-predict_test, Veltri_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: F1-Score; missingness and metric redundancy were penalized.
- Top3: C_AMPs-predict, amplify_bal, pepnet_standard

## Round 39

- Sampled datasets: C_AMPs-predict_test, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: amplify_imb, C_AMPs-predict, AMPsorter

## Round 40

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: amplify_imb, C_AMPs-predict, AMPsorter

## Round 41

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, HMD-AMP

## Round 42

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, AMPsorter

## Round 43

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Precision in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, HMD-AMP

## Round 44

- Sampled datasets: C_AMPs-predict_test, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, amplify_bal, ascan2

## Round 45

- Sampled datasets: ProteoGPT_all_predictions, ProteoGPT_all_predictions, C_AMPs-predict_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, pepnet_standard

## Round 46

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, pepnet_standard

## Round 47

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: NPV; missingness and metric redundancy were penalized.
- Top3: pepnet_standard, HMD-AMP, C_AMPs-predict

## Round 48

- Sampled datasets: Veltri_test, Veltri_test, Veltri_test
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: ascan2, HMD-AMP, pepnet_standard

## Round 49

- Sampled datasets: ProteoGPT_all_predictions, Veltri_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports Specificity in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: ECE; missingness and metric redundancy were penalized.
- Top3: HMD-AMP, pepnet_standard, AMPsorter

## Round 50

- Sampled datasets: ProteoGPT_all_predictions, C_AMPs-predict_test, ProteoGPT_all_predictions
- Statistics expert: Bootstrap evidence most strongly supports NPV in this round.
- Screening expert: Model scores were recomputed with rank normalization so metric scales cannot dominate the decision.
- Reviewer: Largest weight revision: AUROC; missingness and metric redundancy were penalized.
- Top3: AMPsorter, amplify_imb, pepnet_standard
