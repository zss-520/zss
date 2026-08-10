# Literature Meeting Agent Evaluation

- Evaluation version: 1.0
- Attribution status: `current_state_proxy_without_pre_discussion_snapshot`
- Gold models: 8 (valid=5, invalid=3)

## Full meeting screening census

The following counts are meeting/gate decisions over all unique retrieved model identities; only the independent gold subset is an externally audited accuracy estimate.

| Category | Count | Ratio of retrieved |
|---|---:|---:|
| Total unique models retrieved | 487 | 1.0 |
| Valid main benchmark candidates | 59 | 12.11% |
| Rejected or held in total | 428 | 87.89% |
| Misretrieval or out of scope | 209 | 42.92% |
| AMP-relevant but not deployable | 185 | 37.99% |
| Manual review required | 34 | 6.98% |

## Core metrics

| Metric | Value |
|---|---:|
| valid_model_retrieval_recall | 1.0 |
| valid_model_retention_rate | 1.0 |
| wrong_model_detection_recall | 1.0 |
| wrong_model_leakage_rate | 0.0 |
| meeting_screen_precision | 1.0 |
| meeting_screen_accuracy | 1.0 |
| meeting_screen_mcc | 1.0 |
| discussion_filter_yield | 0.375 |
| exclusion_reason_traceability | 1.0 |
| primary_metadata_field_accuracy | 1.0 |
| final_audited_precision | 1.0 |
| final_deployment_contamination_rate | 0.0 |

## Audited decisions

| Model | Gold | Retrieved | Decision | Correct | Final | Reason |
|---|---|---:|---|---:|---:|---|
| AMP Scanner v2 | eligible_main_amp_binary | True | accept | True | True | passed_strict_main_amp_deployment_gate |
| C_AMPs-predict | eligible_main_amp_binary | True | accept | True | True | passed_strict_main_amp_deployment_gate |
| UniproLcad | eligible_main_amp_binary | True | accept | True | True | passed_strict_main_amp_deployment_gate |
| ACEP | eligible_main_amp_binary | True | accept | True | True | passed_strict_main_amp_deployment_gate |
| iAMP-DL | eligible_main_amp_binary | True | accept | True | False | passed_strict_main_amp_deployment_gate |
| EIPpred | ineligible_main_amp_binary | True | reject | True | False | benchmark_candidate_false |
| Allopipe | ineligible_main_amp_binary | True | reject | True | False | benchmark_candidate_false |
| AMP | ineligible_main_amp_binary | True | reject | True | False | ambiguous_generic_model_identity |

## Full per-model meeting decisions

| Model | Decision | Category | Final | Reason |
|---|---|---|---:|---|
| ADAM | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| ADP3 | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| alphafold3 | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| amp-gan | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| AMP_mining_pipeline_Shanxi_vinegar | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| AMPDeep | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| AmpGram | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| ANIA | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| APD | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| APD prediction tool | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| APD2 prediction | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| APD6 Predictor | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| APEX | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| BERT-Protein | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| cAMPs-pred | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| CD-HIT web server | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| CNN-based PDA mechanochromic fingerprint AMP identification | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| CTCM-Neo & ConformaX-PEP | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| DBAASPv3.0 | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| de novo designed bifunctional AMP deep learning model | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| De novo designed bifunctional antimicrobial peptide DL model | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| Deep learning hybrid model (unnamed) | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| Deep learning regression model for antimicrobial peptide design (Witten & Witten 2019) | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| DeepMAMP | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| dPABB | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| In silico AMP design from cuttlefish database (Houyvet et al.) | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| LLAMP | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| Multi-label WKnn-MLR | reject_or_hold | manual_review_required | False | insufficient_task_metadata |
| Multiple DL models reviewed (e.g., AMP-BERT, Deep-AmPEP30, etc.) | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| Mutator | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| panCleave | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| PepMCP | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| SeqGAN-BERT-MLP AMP identifier (Cao et al. 2023) | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| The Antimicrobial Peptide Pipeline | reject_or_hold | manual_review_required | False | amp_relevance_present_but_binary_task_identity_unclear |
| 2020-peptidomics | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| A-CaMP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AAGP | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| ACP-DL | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| ACP-ESM | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| acp-ope | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| aCPP-QSAR model | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| ACPred | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| ADAM_web_server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| ADAM (webserver) | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| ADMETlab 3 | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| AFP-AE-CAE | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AFP_DL | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AFP_DL-QSARES | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AFPtransferPred | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AI4AFP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AI4AVP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AI4AVP_predictor | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AIGCRS-AMP30 | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AIPAMPDS | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| allenCCF | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AllerCatPro | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| AllergenFP | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| AllerTop | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| Allopipe | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| AlphaFold | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| AMP | reject_or_hold | misretrieval_or_out_of_scope | False | ambiguous_model_identity_or_database_platform |
| AMP-BERT GitHub repository | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| amp_de_novo_design_cdGAN | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMP-Designer | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMP-Diffusion | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMP-GPT | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMP-researchprotein | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMP-RL | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMP-RNNpro web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMP target specificity Siamese network | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMP toxicity prediction code | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMP toxicity prediction model (hybrid) | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| AMP0 webserver | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMPA web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMPBenchmark | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMPCLGPT | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AmPEP web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AmPepGen | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMPer web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMPfun | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMPGAN | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMPGAN v2 | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMPGAN v3 | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMPGen | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMPGenix | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMPGP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AmpGPT2 | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AmpGram R package | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMPlify GitHub | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AmpLyze | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMPs-Net | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AMPScanner vr.2 web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMPScanner vr.2 web server (alternate) | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AMPSorter | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| ampsphere | reject_or_hold | misretrieval_or_out_of_scope | False | ambiguous_model_identity_or_database_platform |
| ampsphere_web_server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| amyAMP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| ANIA_github | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Anti_Cp | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Anti_Cp.git | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| anti-flavi | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| Anti-Hepatitis Peptides predictor (ref [9]) | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AntiBP3 GitLab | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AntiBP3 PyPI | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| AntiBP3 Web Server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Anticancer-Peptides-CNN | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| AntiCP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AntiCP 2.0 | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AntiCP2.0 | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| Antifp | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| Antimicrobial | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Antimicrobial Peptide Scanner vr.2 web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Antimicrobial-Peptides | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| APD3 | reject_or_hold | misretrieval_or_out_of_scope | False | ambiguous_model_identity_or_database_platform |
| ApexAmphion | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| ApexGO | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| ARCADIAMP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| aro | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| ATP-Program | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| AVCpred | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| AVP-IC50Pred | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| AVP-predictor | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| Bacterial Wars | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| BACTIBASE | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| BAGEL3 | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| BAGEL4 | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| BERT-AmPEP60 | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| BioAMPify | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| BioPepPred-DLEmb | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| C. acnes-targeted AMP generation pipeline (activity classifier) | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| CalcAMP | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| CalcAMP GitHub repository | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| CAMP | reject_or_hold | misretrieval_or_out_of_scope | False | ambiguous_model_identity_or_database_platform |
| CAMP-RL | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| CAMPR3 | reject_or_hold | misretrieval_or_out_of_scope | False | ambiguous_model_identity_or_database_platform |
| CancerGram | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| CancerPPD2 | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| CatBoost AMP predictor | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| cdGAN | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| CDPfold | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| CellPPD | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Co-AMPpred GitHub repository | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| CoAMPpred | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| COGclassifier | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| COMPASS database | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| ConsAMPHemo | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| CPPMechPred | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| cvfs_hfe | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| DBAASP | reject_or_hold | misretrieval_or_out_of_scope | False | ambiguous_model_identity_or_database_platform |
| dbAMP | reject_or_hold | misretrieval_or_out_of_scope | False | ambiguous_model_identity_or_database_platform |
| dbAMP 3.0 web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| DDM GitHub | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Deep-AmPEP30 web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| deep_AMPpred | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Deep attention based variational autoencoder | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| Deep-AVPiden | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| deep-belief-network | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Deep learning-based AMP design for oral pathogens | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| DeepAFP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| deepAMP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| DeepSeaQuence_biofilms | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| DL-QSARES | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| DLFea4AMPGen | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| E-CLEAP GitHub repository | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| EBAMP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| EIPpred | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| EnDL-HemoLyt | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| esm | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| ESM2-AFPpred | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| FMT-MetagenomicData | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| FungiGuard | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| Generative AMP pipeline (VINCI) | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| Generative approach for precision antimicrobial peptide design | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| GPR-based antifungal peptide selectivity predictor | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| HAPPENN | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| HemoPred | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| HMAMP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| HydraAMP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| iACP | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| iAFP-fLRM | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| iblapps | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Insect VGSC inhibitor prediction model | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| InversePep | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| kneaddata | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Lab | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| learning_sequence_motifs | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| LightGBM | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| LinearDisplay | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| LMPred_AMP_Prediction | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| LSTM-based AMP design model | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| LysePred | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| macrel2020benchmark | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| MAPLE GitHub | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| MAPLE GitHub repository | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| MBC-attention | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| MetagenomicDC | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| MetaPepticon | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| MIC prediction ensemble model (BiLSTM-CNN-MBM) | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| ML-guided directed evolution for AMP development | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| MoFormer | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| msaconverter | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| MSCMamba | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Multifunctional AMP Design Framework (FBGAN-enhanced) | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| nov-fams-pipeline | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Npx | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| NSGA-II-GRU AMP designer | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| PandoraGAN | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| PC6-protein-encoding-method | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| PepCVAE | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| PepForge | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| PepGen 1.0 | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| PepGen 1.0 web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| PepNet web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| PepProtGraphAnalyzer | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| PeptideRanker | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| PepVAE | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| phy | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| PLUM | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| PLUM GitHub | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| PPGC-DVAE | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| PrefixProt | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| PrMFTP | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| PRRSV-AVPeP-ML-Omics | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| Red Sea anticancer peptide SVM model | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| Sequential Properties RNN model | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| shap | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| soft-neighbors-supported-clustering | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| SSEL-CPP | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| StackEnPred | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| STAMP | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| StarPep | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Top-ML | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| ToxIBTL | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| ToxinPred | reject_or_hold | misretrieval_or_out_of_scope | False | explicit_scope_or_task_exclusion |
| TP-LMMSG | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| TransDecoder | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Two_Level_Ensemble-classifier-chain | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Two-stage AVP prediction framework | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| UniAMP web server | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Unnamed CVAE-diffusion AMP generator | reject_or_hold | misretrieval_or_out_of_scope | False | non_binary_or_secondary_peptide_task |
| Urchin | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| Venomics artificial intelligence | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| VirSorter2 | reject_or_hold | misretrieval_or_out_of_scope | False | no_main_amp_binary_task_evidence |
| 3D structure-based AMP activity prediction model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ABP-Finder | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ADAPT | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ADAPTABLE | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AExOp-DCS-SEQ | reject_or_hold | relevant_but_not_deployable | False | failed_deployment_readiness_gate |
| AGRAMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AI-driven AMP motif analysis framework | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Akbar et al. ensemble predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Al-Omari 2024 AMP prediction model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMAP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-CapsNet | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-CLIP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-Detector | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-Distillation | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-DualTransnet | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-FreqNet | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-GSM | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-META | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-MIC | reject_or_hold | relevant_but_not_deployable | False | failed_deployment_readiness_gate |
| AMP MIC predictor (CNN/RNN) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP prediction by multidimensional feature embedding | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP prediction ML model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP prediction server (biosino) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP prediction SVM-LZ | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP prediction tool for aquaculture industries (Gautam et al. 2016) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-RNNpro | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP scanner v.2 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP screening model based on LSTM with attention | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP-SEMiner | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMP0 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPA | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPActiPred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AmpClass | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPDiscover | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AmPEP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPER | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AmpHGT | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPidentifier | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPlify_bal | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPlify_imbal | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| amppred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPpred-AAIW | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPpred-DLFF | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPpred-EL | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPpred-MFA | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPPRED15 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPpredictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPpredMFA | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AMPTrans-lstm | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AniAMPpred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ANN-based AMP prediction model (ref [4]) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ANN-based AMP prediction model (Torrent et al. 2011) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Antibacterial Peptide Binary Classifier (XGBoost) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| antibp | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AntiBP2 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Antimicrobial Peptide Pipeline | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AntiMPmod | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AntiTbPred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AntiVPP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AP_Sin | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| APD (Antimicrobial Peptide Database) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| APD Prediction | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| APEX 1.1 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Appred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| APSvr.2 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AVPIden | reject_or_hold | relevant_but_not_deployable | False | failed_deployment_readiness_gate |
| AVPpred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| AxPEP3 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Bacillus_AMP_DL_models | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Bacteria-specific ML models for E. coli AMP activity | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| BAGEL2 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| BERT-based AMP prediction model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| BERT-based AMP recognition model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| BERT-GRU | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| BERT-TextCNN-based AMP recognition tool | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Bidirectional LSTM AMP classification model (Wang2021) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Bioproteom AAC AMP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| C-PAmP classifier | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CAmidPred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CAMP (Collection of Antimicrobial Peptides) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CAMP database | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CAMPER | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CAMPR3(RF) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CAMPR3(SVM) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CAMPR34 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CAMPR4 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CAST | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CF-AMP prediction | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CG-AMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Chang et al. AVP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Characterization and Identification of Natural Antimicrobial Peptides on Different Organisms | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ChatGPT-based AMP classifier | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CL-ACP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ClassAMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CLASSAMP5 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CNN-based AMP hemolytic activity predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Collaborative Filtering and Link Prediction model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| COMDEL | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Computational AMP prediction from proteomes (Monsalve et al.) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CSAMPPRED | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| CTCM-Neo & ConformaX-PEP framework | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| DBAASP APP tool | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| DBAASP microbial strain-specific AMP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| DBAASP_MSS_AMP_predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| DBAASP v3 prediction tools | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| DBAASP6 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| dbAMP 2.0 AMP scanning tool | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Decision tree model for antimicrobial peptide activity prediction | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Deep-ABPpred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Deep-AmPEP30 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Deep-AVPpred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Deep learning model for AMP discovery from ruminant gastrointestinal microbiomes | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Deep2Pep | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| DeepPepQSAR | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| DMAMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| EFC-FCBF | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Embedded-AMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ENNAVIA | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Ensemble-AMPPred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ESMFold + ESM-2 graph deep learning AMP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Evolutionary feature weighting approach for AMP classification | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| FIRM-AVP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Gabere&Noble AMP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| GAC-BiTCNN-AMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| GAC-BTCNN-Pred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Host defense peptide selectivity Random Forest model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| hydramp | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| iAMP-2L | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| iAMP-CA2L | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| IAMPE | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| iAMPpred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| iAMPred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| iDPF-PseRAAAC | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| iDVIP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ISCAPE | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| LABAMPs | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| LABAMPsGCN | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Linear cationic AMP prediction method | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| LM_pred (BFD) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| LSTM-based AMP classifier/generator | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Macrel | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Malebary-Khan AMP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| MCL-AMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Meta-iAVP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| MLAMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| MLBP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Multi-label weighted KNN-MLR model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| MultiPep | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Multiple alignment based AMP predictor (ref [5]) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| PCSPred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Pep-CNN | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| PepAnno | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Peptide_Predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Plant_AMP_XGBoost_framework | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| PLSR-based α/β-peptide activity predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| PPTPP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Predictive and Interpretable ML Models | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ProtBert-NN AMP activity predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| ProteinBERT (for AMP classification) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| RF-AmPEP30 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| RF-based AMP prediction model (Wani et al. 2021) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Rough set-based AMP prediction model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| sAMP-PFPDeep | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| sAMP-VGG16 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| SenseXAMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Sequence alignment-SVM-LZ complexity model (ref [8]) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| SMEP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| SMO-based lantibiotic predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Sparse Neural Network Models of Antimicrobial Peptide-Activity Relationships | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| StaBle-ABPpred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| StackAMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| StackDPPred | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| StM | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Target-AMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Thakur et al. AVP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Thomas et al. 2009 AMP prediction model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Two-level fuzzy K-NN model (ref [7]) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| UniAMP | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| UniDL4BioPep | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Unnamed AMP predictor from DRAMP 2.0 | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| VEIP prediction model | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Wang et al. AMP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Witten&Witten AMP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| XGBoost AMP prediction model (Bhangu2025) | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| Zare et al. AVP predictor | reject_or_hold | relevant_but_not_deployable | False | missing_code_repository |
| 2022-iAMP-DL | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| ACEP | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| AEPMA | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| AI4AMP | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| AMP-BERT | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| AMP-EF | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| AMP Scanner | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| AMP Scanner v2 | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| AMP-zGSM | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| AMPBAN | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| amPEPpy | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| AMPfinder | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| Ampir | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| AMPlify | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| AMPml | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| AMPSpeciesSpecific | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| AntiBP3 | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| APEX predictor | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| APIN | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| AxPEP | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| BBATProt | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| BPFun | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| C_AMPs-predict | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| CELA-MFP | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| ClaAMP | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| CLABP | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| Co-AMPpred | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| CS-AMPpred | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| CVAE-BIO | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| Cysmotif searcher pipeline | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| DDM | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| E-CLEAP | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| esm-AxP-GDL | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| ExtraTree-based AMP classifiers | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| HMD-AMP | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| iAMP-bert | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| iAMP-DL | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| iAMP-SeE | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| iAMPCN | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| iMFP-LG | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| LMPred | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| MAPLE | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| multiAMP | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| PepNet | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| PGAT-ABPp | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| Pore-Forming_AMP_SVM | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| PreAMP | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| PyAMPA | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| rAMPage | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| SA-MTP | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| SAMP | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| sAMPpred-GAT | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| SGAC | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| smAMPsTK | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| SSFGM-Model | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| TriNet | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| TriStack | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |
| UniproLcad | accept | valid_main_benchmark_candidate | True | passed_strict_main_amp_deployment_gate |
| WeightedEnsemble_L3 (Anti_Cp) | accept | valid_main_benchmark_candidate | False | passed_strict_main_amp_deployment_gate |

> These metrics evaluate the current end-to-end literature pipeline. Strict causal measurement of meeting value-added requires immutable pre-discussion and post-discussion snapshots for each run.
