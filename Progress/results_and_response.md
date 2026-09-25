# Results and response status
Manuscript NEURI-D-26-00231 (Neuroscience Informatics) - resubmission due 7 Oct 2026

The responses in Section 3 assume the manuscript edits listed under each one are made. All numbers come from the executed notebook.

## 1. Protocol
- **Data:** pre-extracted feature file, 81,426 windows, 924 features (22 channels x 42), class-balanced by random undersampling when the file was built.
- **Fixed split (by patient):** train chb01-chb12 (62,182 windows), validation chb14-chb15 (5,750), test chb16, chb17, chb18, chb19, chb20, chb22 (13,494; 55.3% seizure). chb17a/b/c are one patient; chb21 is grouped with chb01 (same patient).
- **LOSO:** 20 patients, each held out once. Imputation, scaling, training and threshold selection never use the held-out patient. XGBoost with default (untuned) hyperparameters.
- **Metrics:** ROC-AUC is threshold-free; sensitivity and specificity use a fixed threshold of 0.5 (nothing tuned).

## 2. Results

### 2.1 Headline
For reference, the manuscript's random segment split gave ROC-AUC 0.998 and 98.81% accuracy; this is now considered inflated by leakage.

| Setting | Model | ROC-AUC | PR-AUC | Sens @0.5 | Spec @0.5 |
|---|---|---|---|---|---|
| 6 unseen test patients | Logistic regression | 0.727 | 0.800 | 0.578 | 0.785 |
| 6 unseen test patients | XGBoost, defaults | 0.796 | 0.848 | 0.482 | 0.932 |
| 6 unseen test patients | XGBoost, Optuna-tuned | 0.803 | 0.853 | 0.540 | 0.904 |
| LOSO, 20 patients (mean ± SD) | XGBoost, defaults | 0.781 ± 0.196 | 0.792 ± 0.248 | 0.573 ± 0.254 | 0.819 ± 0.168 |

On the test set a useless model scores PR-AUC 0.553 and 55.3% accuracy (always predicting seizure), so accuracy is not used as the headline.

### 2.2 LOSO across the 20 patients (mean ± SD, 95% bootstrap CI over patients)
| Metric | Mean ± SD | 95% CI |
|---|---|---|
| ROC-AUC | 0.781 ± 0.196 | 0.691-0.859 |
| PR-AUC | 0.792 ± 0.248 | 0.677-0.889 |
| Sensitivity @0.5 | 0.573 ± 0.254 | 0.464-0.679 |
| Specificity @0.5 | 0.819 ± 0.168 | 0.745-0.887 |
| Balanced accuracy @0.5 | 0.696 ± 0.137 | 0.635-0.753 |
| Precision @0.5 | 0.745 ± 0.294 | 0.612-0.863 |
| F1 @0.5 | 0.613 ± 0.246 | 0.502-0.713 |

### 2.3 LOSO per patient (threshold 0.5)
| Patient | Windows | Seizure % | ROC-AUC | Sens | Spec | Bal. acc | Acc (derived) |
|---|---|---|---|---|---|---|---|
| chb01 | 6,051 | 57.9% | 0.862 | 0.459 | 0.973 | 0.716 | 0.675 |
| chb02 | 1,102 | 54.6% | 0.891 | 0.623 | 0.996 | 0.809 | 0.792 |
| chb03 | 3,090 | 57.8% | 0.859 | 0.301 | 0.979 | 0.640 | 0.587 |
| chb04 | 6,127 | 35.9% | 0.905 | 0.441 | 0.994 | 0.717 | 0.795 |
| chb05 | 3,498 | 72.2% | 0.951 | 0.882 | 0.871 | 0.876 | 0.879 |
| chb06 | 9,288 | 13.0% | 0.536 | 0.449 | 0.570 | 0.510 | 0.554 |
| chb07 | 4,255 | 43.6% | 0.948 | 0.864 | 0.911 | 0.888 | 0.891 |
| chb08 | 6,302 | 79.2% | 0.826 | 0.582 | 0.906 | 0.744 | 0.649 |
| chb09 | 3,843 | 32.7% | 0.932 | 0.946 | 0.523 | 0.735 | 0.661 |
| chb10 | 6,480 | 42.8% | 0.869 | 0.764 | 0.783 | 0.774 | 0.775 |
| chb11 | 4,543 | 89.9% | 0.944 | 0.931 | 0.718 | 0.824 | 0.909 |
| chb12 | 7,603 | 48.3% | 0.593 | 0.580 | 0.562 | 0.571 | 0.571 |
| chb14 | 3,606 | 40.2% | 0.310 | 0.249 | 0.492 | 0.370 | 0.394 |
| chb15 | 2,144 | 62.8% | 0.347 | 0.008 | 0.909 | 0.458 | 0.343 |
| chb16 | 2,688 | 30.6% | 0.634 | 0.366 | 0.870 | 0.618 | 0.716 |
| chb17 | 2,453 | 69.1% | 0.895 | 0.328 | 0.992 | 0.660 | 0.533 |
| chb18 | 2,451 | 61.7% | 0.890 | 0.549 | 0.967 | 0.758 | 0.709 |
| chb19 | 1,435 | 65.8% | 0.782 | 0.650 | 0.815 | 0.733 | 0.706 |
| chb20 | 3,060 | 52.5% | 0.727 | 0.626 | 0.736 | 0.681 | 0.678 |
| chb22 | 1,407 | 62.8% | 0.930 | 0.865 | 0.818 | 0.842 | 0.848 |

Accuracy is derived from the rounded sensitivity, specificity and seizure fraction. It depends strongly on the seizure fraction (13% to 90%), so ROC-AUC and balanced accuracy are the fairer per-patient comparison.

### 2.4 SNN ablation (LOSO, same folds, XGBoost defaults, paired per-patient AUC differences vs. conventional only)
| Feature set | Mean AUC ± SD | 95% CI | Difference vs conventional | 95% CI of difference | Wilcoxon | Better in |
|---|---|---|---|---|---|---|
| Conventional (34/channel) | 0.784 ± 0.195 | 0.695-0.861 | - | - | - | - |
| Spike features only (8/channel) | 0.628 ± 0.108 | 0.582-0.675 | -0.156 | -0.212 to -0.093 | p < 0.001 | 3/20 patients |
| All features (42/channel) | 0.781 ± 0.196 | 0.691-0.859 | -0.002 | -0.009 to +0.004 | p = 0.50 | 9/20 patients |

## 3. Concerns that are fixed

### Reviewer 1, comment 1 - data leakage from overlapping windows
*Windows with 90% (seizure) and 50% (interictal) overlap were generated before a random split, so near-identical windows can sit in different partitions. Split at patient, recording or event level (or another grouped strategy) and report the main results again.*

**Response**
- We agree: the random segment-level split let overlapping windows fall into different partitions.
- We replaced it with a patient-level split, so every window of a patient stays in one partition (train chb01-chb12, validation chb14-chb15, test chb16-chb22; chb17a/b/c and chb21/chb01 treated as single patients).
- Imputation and scaling are fitted on the training patients only.
- On the six unseen test patients XGBoost reaches ROC-AUC 0.796 (defaults) and 0.803 (Optuna-tuned), in line with the patient-grouped cross-validation on the training patients (0.815).
- The earlier random split gave ROC-AUC 0.998 and 98.81% accuracy; we regard it as inflated and no longer use it as the main result.
- At a fixed threshold of 0.5 the accuracy is 68.3% (always predicting seizure gives 55.3%), so we headline ROC-AUC instead of accuracy.
- Because the split is by patient, no overlapping or adjacent windows can cross partitions.

*Edit needed: abstract, Sections 2.1, 3, 4 and 5 (replace the random-split results; remove the leakage item from future work).*

### Reviewer 1, comment 2 - complete LOSO evaluation
*Report per-subject results, mean and SD, sensitivity, specificity, precision, F1, ROC-AUC and confidence intervals, and clarify that balancing, tuning, training and threshold selection are done independently within each fold.*

**Response**
- We ran full Leave-One-Subject-Out over 20 patients (chb17a/b/c merged, chb21 grouped with chb01), each held out once.
- Per-patient results are in Section 2.3 and mean ± SD with 95% bootstrap CIs across patients in Section 2.2.
- ROC-AUC is 0.781 ± 0.196 (95% CI 0.691-0.859; range 0.31-0.95). At a fixed 0.5: sensitivity 0.573 ± 0.254, specificity 0.819 ± 0.168, precision 0.745 ± 0.294, F1 0.613 ± 0.246.
- In every fold the imputer, scaler and model use the training patients only, and any threshold is chosen on two other patients, never the held-out one; the headline numbers use ROC-AUC and a fixed 0.5.
- Hyperparameters were the untuned XGBoost defaults, so no held-out patient influenced model selection.
- Class balancing (random undersampling of the interictal class) was done once when the feature file was built, before splitting, so it is not repeated inside each fold; we state this as a limitation.

*Edit needed: replace the single 89.62% figure with Sections 2.2 and 2.3 as a new table; add the balancing limitation to Section 4.*

### Reviewer 1, comment 3 - contribution of the LIF/SNN features
*Add an ablation (conventional only, LIF/SNN only, combined) with the same classifier, split and procedure, and report confidence intervals or significance.*

**Response**
- We ran the ablation under the same LOSO folds, classifier (XGBoost defaults) and procedure: conventional only (34 features/channel), spike features only (8/channel) and all 42/channel.
- Mean ROC-AUC over the 20 patients: 0.784 conventional, 0.628 spike only, 0.781 combined (Section 2.4).
- Adding the spike features to the conventional ones changes AUC by -0.002 (paired, 95% CI -0.009 to +0.004; Wilcoxon p = 0.50; better in 9 of 20 patients).
- Spike features alone are above chance (0.628, CI 0.582-0.675) but 0.156 below conventional features (CI -0.212 to -0.093; p < 0.001).
- On this dataset the LIF/SNN features therefore carry weak signal but no measurable complementary information, consistent with their lowest SHAP attribution.
- We changed the claim from a central contribution to an exploratory representation.
- This also answers the SNN-block ablation requested by Reviewer 2 (comment 4).

*Edit needed: abstract, contribution 4 in Section 1, Section 2.3.1 ("what a real neuron would infer"), Section 3 and the conclusion.*

### Reviewer 2, comment 6 - LOSO reported only as a single mean
*Per-subject accuracies, standard deviation and ideally confidence intervals are needed to judge the reliability of cross-patient transfer.*

**Response**
- Per-subject results for all 20 patients are now reported (Section 2.3), with SD and 95% bootstrap confidence intervals (Section 2.2).
- Mean ROC-AUC is 0.781 ± 0.196 (95% CI 0.691-0.859); 12 of 20 patients reach 0.85 or more and 5 fall below 0.70 (chb06, chb12, chb14, chb15, chb16).
- Per-patient accuracy at the fixed 0.5 threshold is listed but depends strongly on each patient's seizure fraction (13% to 90%), so we also report balanced accuracy (0.696 ± 0.137), sensitivity and specificity.
- Sensitivity and specificity vary widely between patients, so the mean alone would hide unreliable transfer; we report the full spread.
- The earlier single LOSO figure (89.62%) is replaced by these recomputed results.

*Edit needed: Section 3 (new LOSO table) and the discussion of the subject-mixed vs. LOSO gap.*

### Reviewer 3 - leakage and unclear contribution of the LIF/SNN features
*Highly overlapping segments and segment-level random splitting may inflate accuracy and AUC; the contribution of the LIF/SNN features is unclear without ablation.*

**Response**
- We agree that segment-level random splitting with overlapping windows can inflate accuracy and AUC.
- All main results now come from patient-level evaluation: a fixed split with unseen test patients and full LOSO (Sections 2.1-2.3).
- Under it, ROC-AUC falls from 0.998 (random split, previous manuscript) to 0.796-0.803 on unseen test patients and 0.781 ± 0.196 in LOSO; accuracy is no longer the headline.
- We ran the missing ablation: adding LIF/SNN features to the conventional features changes AUC by -0.002 (95% CI -0.009 to +0.004), so their added contribution is not supported (see Reviewer 1, comment 3).
- The claims in the abstract and conclusion are revised to match these results.

*Edit needed: same places as Reviewer 1, comments 1 and 3.*

## 4. Not fixed yet
- **Reviewer 1:** 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
- **Reviewer 2:** 1, 2, 3, 4, 5, and the minor comments
- **Reviewer 3:** the third point (claims about clinical use and biological interpretation)
- **Editor:** point-by-point response letter, source files (LaTeX), journal formatting

Analysis settled but manuscript text still to change: R1-4, R1-5, R2-5. Partly done: R1-12, R2-4 (only the SNN block was ablated), R2-6 (the paediatric-only and binary-formulation limitations are still to discuss).
