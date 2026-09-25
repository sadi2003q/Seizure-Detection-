# Reviewer concerns and responses
Manuscript NEURI-D-26-00231 (Neuroscience Informatics) -- resubmission due 7 Oct 2026

Concerns are copied from the editor and reviewer letter. Where no response has been produced yet, the response section says so rather than being left blank.

## Editor -- Decision letter

**Concern**

> The reviewers recommend reconsideration of your manuscript following major revision. Please outline every change made in response to their comments and provide suitable rebuttals for any comments not addressed. Please resubmit your revised manuscript by Oct 07, 2026.

**Response**

Not covered yet.

## Reviewer 1 -- Comment 1 -- data leakage from overlapping windows

**Concern**

> Please address the possible data leakage caused by overlapping EEG windows. Seizure-associated segments are generated with 90% overlap, while interictal segments use 50% overlap. If these overlapping windows are generated before the random train, validation, and test split, highly similar windows can appear in different partitions. I recommend addressing this in the current study rather than leaving it for future work. Please perform the split at the patient, recording, or seizure-event level before generating overlapping windows, or use another grouped splitting strategy that prevents overlapping or adjacent windows from crossing partitions. The main performance results should then be reported again using this leakage-controlled setup.

**Response**

FIXED. Replaced the random segment-level split with a patient-level split, so every window of a patient stays in one partition (train chb01-chb12, validation chb14-chb15, test chb16-chb22; chb17a/b/c and chb21/chb01 treated as single patients). Imputation and scaling are fitted on the training patients only. On the six unseen test patients, XGBoost reaches ROC-AUC 0.796 (defaults) and 0.803 (Optuna-tuned), in line with the patient-grouped cross-validation on the training patients (0.815). The earlier random split gave ROC-AUC 0.998 and 98.81% accuracy; this is now considered inflated by leakage and is no longer the main result.

## Reviewer 1 -- Comment 2 -- complete LOSO evaluation

**Concern**

> Please report the results for individual subjects in addition to the overall value. Mean and standard deviation across subjects would be useful, along with sensitivity, specificity, precision, F1-score, and ROC-AUC where possible. Confidence intervals would further strengthen the analysis. Please also clarify whether class balancing, hyperparameter optimization, model training, and threshold selection are performed independently within each LOSO fold. The held-out subject should not contribute information to any model-selection or preprocessing step.

**Response**

FIXED for the model-selection/preprocessing independence and per-subject reporting; PARTLY FIXED for class balancing. Full LOSO over 20 patients, each held out once. Per-patient results (n, seizure fraction, ROC-AUC, sensitivity, specificity, balanced accuracy) plus mean +/- SD and 95% bootstrap CI across patients: ROC-AUC 0.781 +/- 0.196 [0.691, 0.859], sensitivity 0.573 +/- 0.254, specificity 0.819 +/- 0.168 at a fixed threshold of 0.5. In every fold, the imputer, scaler, model, and threshold are fitted using only the training patients; the held-out patient contributes nothing to preprocessing or model selection. RESPONSE STILL NEEDED: class balancing (random undersampling of the interictal class) was performed once when the feature file was built, before any split, so it is not repeated independently within each LOSO fold; we state this as a limitation rather than having fixed it.

## Reviewer 1 -- Comment 3 -- contribution of the LIF/SNN features

**Concern**

> The SHAP analysis shows that the Spike/SNN domain has the lowest aggregate attribution among the five feature domains. I recommend adding an ablation study comparing: (1) conventional EEG features without LIF/SNN features, (2) LIF/SNN features alone, and (3) conventional EEG features combined with LIF/SNN features. Please use the same classifier, data split, and evaluation procedure for these comparisons. Reporting confidence intervals or statistical significance for the improvement would help establish whether the LIF/SNN features provide a meaningful contribution.

**Response**

FIXED. Ran the ablation under the same LOSO folds, classifier (XGBoost defaults), and procedure. Mean ROC-AUC over 20 patients: conventional only 0.784, spike-only 0.628, all features combined 0.781. Adding spike features to the conventional set changes AUC by -0.002 (paired, 95% CI -0.009 to +0.004; Wilcoxon p = 0.50; better in 9 of 20 patients) -- consistent with no measurable contribution. Spike features alone are above chance but clearly weaker than conventional features (difference -0.156, 95% CI -0.212 to -0.093, p < 0.001). We changed the claim from a central contribution to an exploratory representation.

## Reviewer 1 -- Comment 4 -- verify the reported performance values

**Concern**

> There are several inconsistencies in the numerical results. The main subject-mixed stacked ensemble accuracy is reported as 98.81%, while later sections describe a reduction from 97.81% to 89.62% under LOSO. Table 9 reports non-seizure recall as 0.9787, while the paragraph immediately below reports 0.9586. There is also a difference between the stacked ensemble F1-score of 0.9855 in Table 8 and the macro/weighted F1-score of 0.9780 in Table 9. I recommend recalculating the reported metrics directly from the final predictions or confusion matrix and checking the abstract, tables, results, discussion, and conclusion for consistency.

**Response**

NOT YET COVERED. The underlying analysis is settled (all leak-free results in this session were regenerated programmatically from stored predictions, so this class of inconsistency cannot recur), but the manuscript text itself -- abstract, Tables 8/9/12, results, discussion, conclusion -- still has the old 98.81% / 97.81% / 89.62% figures and the Table 9 mismatch, and these have not yet been replaced with the new leak-free numbers.

## Reviewer 1 -- Comment 5 -- decision-threshold selection procedure

**Concern**

> The methodology states that the decision threshold is selected by maximizing validation F1-score. However, Table 10 reports an F1-score of 0.9997 at threshold 0.07 and 0.9990 at threshold 0.11, while 0.11 is reported as the selected threshold. Please provide the complete threshold search and explain the exact selection criterion. Also, please avoid referring to 0.11 as a default threshold.

**Response**

NOT YET COVERED. We now select the threshold by balanced accuracy rather than F1, and headline ROC-AUC together with a fixed threshold of 0.5 rather than a tuned value, because with as few as two validation patients the tuned threshold repeatedly landed at the edge of the search grid (e.g. 0.01 or 0.99) and produced degenerate sensitivity/specificity trade-offs. This is a settled modelling decision, but the manuscript's methodology and results text describing F1-based threshold selection and Table 10 have not yet been rewritten to match it.

## Reviewer 1 -- Comment 6 -- inconsistency in the LIF burst threshold

**Concern**

> Table 5 reports the burst ISI threshold as 40 ms, while the text following the table states 50 ms. Please confirm which value was actually used in the experiments and make the methodology, equations, tables, and implementation description consistent.

**Response**

NOT COVERED. This requires checking the feature-extraction code (outside the modelling pipeline worked on in this session) to confirm which value was actually used, then correcting the manuscript for consistency.

## Reviewer 1 -- Comment 7 -- definition of the positive seizure class

**Concern**

> The seizure annotations are extended by 10 seconds before seizure onset and 5 seconds after seizure offset. Please provide a clear reason for this choice. It may also be more accurate to use the term seizure-associated or ictal-associated segments when referring to these extended windows. An additional experiment using only the original seizure onset and offset annotations would be useful to show how much the annotation extension affects classification performance.

**Response**

PARTLY FIXED. The requested experiment is built and ready to run: a new CSV column (in_core_ictal) lets us compare the extended-window label against the true ictal interval under the same LOSO protocol, same classifier, and a paired per-patient statistical comparison. It has been validated on synthetic data but not yet run on the real dataset, so the actual result and the terminology change to seizure-associated/ictal-associated in the manuscript are still outstanding.

## Reviewer 1 -- Comment 8 -- evaluation using the natural class distribution

**Concern**

> The interictal class is randomly undersampled to create a balanced dataset. I recommend adding an evaluation using the natural class distribution. Measures such as event-level seizure sensitivity, false positives per hour, specificity, and detection latency would provide a more clinically meaningful assessment.

**Response**

NOT COVERED. Blocked: the available feature file is pre-balanced 50/50 by random undersampling before any labels of window timing were retained at the needed granularity; the natural class distribution, false positives per hour, and detection latency cannot be computed from it. Needs the un-balanced feature set (or raw EEG) from the original extraction pipeline.

## Reviewer 1 -- Comment 9 -- moderate physiological and clinical claims

**Concern**

> Statements suggesting that the LIF transformation represents what a real neuron would infer from the EEG signal appear stronger than the evidence currently supports. Describing it as a biologically inspired spike-domain transformation or nonlinear temporal encoding would be more precise. Similarly, statements about clinical viability should be made carefully unless supported by event-level performance, false-alarm analysis, detection latency, and validation under realistic clinical conditions.

**Response**

NOT YET COVERED. The evidence to justify this wording change exists (leak-free sensitivity at a fixed 0.5 misses roughly 43% of seizure windows, and the SNN ablation shows no added contribution), but the manuscript wording itself -- the LIF-interpretation language and any clinical-viability claims -- has not yet been revised.

## Reviewer 1 -- Comment 10 -- comparison with previous seizure-detection studies

**Concern**

> Table 12 compares the proposed method with several recent studies, but the listed methods use different class-balancing strategies, patient-specific settings, datasets or subsets, and evaluation protocols. It would be useful to separate subject-dependent and subject-independent studies, and the LOSO result should be compared primarily with studies that evaluate models on previously unseen subjects.

**Response**

NOT COVERED. Writing task: Table 12 needs to be reorganised into subject-dependent versus subject-independent groups and the LOSO result compared against the latter.

## Reviewer 1 -- Comment 11 -- strengthen the SHAP analysis

**Concern**

> Please consider reporting the most important individual EEG features and their direction of influence on model predictions. Representative SHAP explanations for seizure and non-seizure examples would also improve the interpretation. It would also be interesting to examine whether the most important features remain reasonably consistent across LOSO subjects.

**Response**

NOT COVERED. Not yet built: SHAP needs to be recomputed on the leak-free model (the original SHAP ranking was produced under the leaky random split), including top individual features with direction of effect, example explanations for seizure and non-seizure windows, and a check of consistency across the 20 LOSO folds.

## Reviewer 1 -- Comment 12 -- additional information for reproducibility

**Concern**

> Please include the exact EEG channels used, subject and recording inclusion criteria, random seeds, class undersampling procedure, Optuna search spaces, number of optimization trials, final hyperparameters, cross-validation settings, LOSO procedure, complete threshold search, and all LIF feature-extraction parameters. If possible, providing the preprocessing and model implementation code would further improve reproducibility.

**Response**

PARTLY FIXED. The modelling code (preprocessing, LOSO, ablation, per-model files) is on GitHub and runs on Kaggle, and the LOSO/ablation procedure, seeds, and the patient-level split are fully documented in that repository's README. RESPONSE STILL NEEDED: the exact EEG channels, subject and recording inclusion criteria, the original undersampling procedure, and the LIF feature-extraction parameters live in the upstream feature-extraction script, which has not yet been documented in the manuscript; the Optuna search spaces and trial counts used for the leak-free results also still need to be written up.

## Reviewer 1 -- Comment 13 -- expand the related work on EEG signal representations

**Concern**

> The related-work section could include additional recent studies exploring alternative EEG representations, including time-frequency and wavelet-based approaches, for example Korani & Domakonda (2025) on time-frequency EEG representations.

**Response**

NOT COVERED. Writing task: add the suggested and related literature to Section 1.

## Reviewer 1 -- Comment 14 -- language and consistency review

**Concern**

> Several grammatical and terminology issues need correction (e.g. 42 features items per channel, an example of EEG segment, nonlinear/non-linear, interictal/inter-ictal, LIF/SNN). There is also a contradiction in the segmentation description: windows are initially described as non-overlapping, but seizure-associated and interictal windows are later said to have 90% and 50% overlap respectively.

**Response**

NOT COVERED. Writing/proofreading task across the manuscript text.

## Reviewer 2 -- Comment 1 -- rationale and coherence of the multi-domain feature set

**Concern**

> The manuscript does not adequately justify why these particular descriptors were chosen, how they relate to one another mechanistically, or why they are expected to be complementary rather than redundant. The authors should explicitly engage with the entropy / AR-complexity / spectral-power literature and clarify the expected physiological meaning of the full 42-feature vector.

**Response**

NOT COVERED. Writing task: needs a mechanistic rationale in Section 1/2.2 linking the chosen features to the cited neurophysiology literature.

## Reviewer 2 -- Comment 2 -- spectral and power-related features, methodological transparency

**Concern**

> It is not stated whether the frequency bands are obtained purely by integration of the Welch estimate or whether additional filtering or wavelet-based decomposition is applied, nor is the precise power calculation specified. The authors should justify the choice of Welch + fixed clinical bands and ideally provide a sensitivity analysis or comparison with an alternative spectral estimator.

**Response**

NOT COVERED. Blocked: an alternative spectral estimator comparison requires access to the raw EEG or the extraction pipeline, which is outside the modelling work done in this session.

## Reviewer 2 -- Comment 3 -- limited treatment of entropy/complexity measures

**Concern**

> Only a single permutation-entropy feature appears among the nonlinear descriptors. Expanding or at least discussing the entropy family, and relating the chosen measures to the cited literature, would strengthen the manuscript.

**Response**

NOT COVERED. Blocked: adding alternative entropy estimators requires re-extracting features from raw EEG, which is outside the modelling work done in this session.

## Reviewer 2 -- Comment 4 -- feature combination and potential redundancy

**Concern**

> No correlation analysis, mutual-information ranking, or systematic ablation is reported to demonstrate that the five feature families are complementary rather than largely collinear. An ablation study removing entire domains (especially the novel SNN block) and quantifying the statistical significance of any performance drop is essential.

**Response**

PARTLY FIXED. The SNN-block ablation is done (see Reviewer 1, comment 3): removing it changes AUC by -0.002 (95% CI -0.009 to +0.004), i.e. no measurable drop. RESPONSE STILL NEEDED: ablating the other four domains (time, frequency, wavelet, nonlinear) individually and a feature-level correlation/redundancy analysis have not yet been run.

## Reviewer 2 -- Comment 5 -- hyper-parameter selection and threshold calibration

**Concern**

> The search spaces, number of trials, and final hyper-parameter values are only partially reported. The near-perfect validation metrics across a range of thresholds raise the possibility of residual leakage from the heavy temporal overlap used in the two-pass segmentation; this should be addressed more rigorously, e.g. by a stricter non-overlapping or patient-wise temporal split for threshold selection.

**Response**

PARTLY FIXED. The leakage in threshold selection is addressed: the split is now patient-level and the threshold is chosen on separate validation patients, never on the training or test patients. RESPONSE STILL NEEDED: the exact Optuna search spaces, number of trials, and final hyperparameter values for the leak-free results still need to be written into the manuscript's methodology.

## Reviewer 2 -- Comment 6 -- generalisation and evaluation protocol

**Concern**

> The LOSO result is reported only as a single mean. Per-subject accuracies, standard deviation, and ideally confidence intervals would allow readers to judge the reliability of cross-patient transfer. The paediatric-only nature of CHB-MIT and the binary formulation limit clinical generalisability; these limitations could be discussed more quantitatively.

**Response**

FIXED for the per-subject/SD/CI request; NOT YET COVERED for the paediatric-only and binary-formulation discussion. Per-subject results for all 20 patients are reported with SD and 95% bootstrap confidence intervals: mean ROC-AUC 0.781 (95% CI 0.691 to 0.859); 12 of 20 patients reach 0.85 or more and 5 fall below 0.70 (chb06, chb12, chb14, chb15, chb16). RESPONSE STILL NEEDED: a quantitative discussion of the paediatric-only cohort and the binary seizure/non-seizure formulation as generalisability limitations has not yet been written.

## Reviewer 2 -- Minor comments

**Concern**

> Several tables contain typographical inconsistencies or truncated mathematical expressions (especially Table 6). The LIF parameter choices would benefit from a brief sensitivity analysis. Figure 1 and the pipeline diagram should use high-resolution versions and consistent notation. The comparison table should more carefully note differences in evaluation protocol.

**Response**

NOT COVERED. Table 6 typo fixes and figure quality are a writing/formatting task; the LIF parameter sensitivity analysis is blocked, for the same reason as Reviewer 2 comments 2 and 3.

## Reviewer 3 -- Main comment -- leakage and unclear LIF/SNN contribution

**Concern**

> The primary issue is the possibility of data leakage arising from highly overlapping EEG segments and segment-level random splitting, which may substantially inflate the reported accuracy and AUC. The actual contribution of the proposed LIF/SNN features is unclear without appropriate ablation experiments, and some neurophysiological interpretations appear stronger than what can be justified from scalp EEG data.

**Response**

FIXED for leakage and the LIF/SNN ablation; NOT YET COVERED for the neurophysiological wording. All main results now come from patient-level evaluation (fixed split with unseen test patients and full LOSO); ROC-AUC falls from 0.998 (previous random split) to 0.796-0.803 on unseen test patients and 0.781 in LOSO, and the requested LIF/SNN ablation shows no added contribution over conventional features (see Reviewer 1, comment 3). RESPONSE STILL NEEDED: the manuscript's neurophysiological interpretation and clinical-applicability wording have not yet been revised to match this evidence (same item as Reviewer 1, comment 9).
