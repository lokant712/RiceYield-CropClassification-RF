# MDI3003 Advanced Predictive Analytics — Experiment 08
## Agricultural Predictive Analytics: Rice Yield Prediction with a Guided Crop-Label Classification Extension

**Student:** Lokanth S  
**Registration Number:** 23MID0037  
**Faculty Coordinator:** Dr. Durgesh Kumar  
**Manual Version:** 3.0  
**GitHub Repository:** `lokant712/RiceYield-CropClassification-RF`

---

## 1. Problem Contract & Prediction Scope

- **Target Variable:** Rice yield in metric tonnes per hectare (`t/ha`) for a single district–season–harvest-year observation.
- **Predictor Whitelist:** `state`, `district`, `season`, `year` (constant `REG = ['state', 'district', 'season', 'year']`).
- **Explicit Exclusions & Anti-Leakage Rules:** Harvested area, total production, target-derived ratios, full-season weather summaries, and identifiers encoding yield are strictly excluded. Combining production and area algebraically reconstructs the target ($Yield = Production / Area$), which constitutes severe target leakage.
- **Operational Prediction Setting:** Pre-season regional planning and macro-allocation decision-support utilizing strictly pre-season administrative knowledge.
- **Core Hypothesis:** Multi-variable modeling (Ridge trend regression, Decision Trees, Random Forests) reduces prediction error relative to a historical median baseline.

---

## 2. Data Governance & Provenance Record

### 2.1 Explicit Acquisition Audit (D5 → D1)
In strict accordance with the manual's data acquisition hierarchy:
1. **Candidate D5 (Mendeley Data DOI `10.17632/ywp3y5j9vv.1`):** Ingest audit attempted on the ICRISAT dataset published by Souryabrata Mohapatra (July 11, 2023). While metadata parsed with a CC BY 4.0 license, the deposit contains 0 attached data payloads in the public snapshot (`files: []`). Direct API file endpoints returned 404/401.
2. **Candidate D1 (Government of India APY Database):** Acquired the authentic district-season crop statistics table (`crop_production.csv`).
   - **Authentic SHA-256:** `5b2637058ca4aff2ef7c0e0f2f391bca712300d082693b585f489871fd91d520`
   - **Raw Row Count:** 246,091 records across 33 Indian States/UTs (1997–2015).
   - **Clean Rice Observations:** 15,078 records post non-positive area / physical bounds filtering `[0.0, 15.0]` t/ha.
   - **Target Unit Verification:** Area in hectares (`ha`), Production in tonnes (`t`), Derived Yield = `Production / Area` in tonnes/hectare (`t/ha`).

### 2.2 Classification Extension Dataset (Crop Recommendation)
- **File:** `data/raw/crop_recommendation.csv` (147,833 bytes, SHA-256: `808e1d84a544647f1ab2fdae04f434b1842692d6be09eee18a5c2d43eb78d77c`)
- **Structure:** 2,200 observations, 22 balanced crop classes (100 samples/class), 7 soil/climate features (`N, P, K, temperature, humidity, ph, rainfall`).
- **Documented i.i.d. Rationale (`iid_justified=True`):** Unlike the longitudinal district rice yield panel data requiring chronological partitioning, the crop recommendation task evaluates point-in-time soil physicochemical parameters (N, P, K, pH) and micro-climate readings. Each record is an independent agronomic laboratory/environmental profile. Therefore, random stratified 60/20/20 partitioning is justified and maintains class balance across partitions.

---

## 3. Empirical Results Summary

### 3.1 Regression Validation Performance (1997–2011 Train vs. 2012–2013 Validation)
Models ranked by validation MAE (stable sort preserving simpler models on ties):

| Candidate Model | Validation MAE (t/ha) | Validation RMSE (t/ha) | Validation R² | Validation MedAE (t/ha) | Status |
|---|---|---|---|---|---|
| **`ridge_trend`** | **0.4748** | **0.6678** | **0.5199** | **0.3667** | **Selected Winner** |
| `forest` (RF depth=12) | 0.5761 | 0.7706 | 0.3607 | 0.4533 | Rank 2 |
| `tree` (DT depth=6) | 0.6886 | 0.8742 | 0.1772 | 0.5694 | Rank 3 |
| `median` (Dummy baseline) | 0.8270 | 1.0658 | -0.2230 | 0.6994 | Predeclared Baseline |

### 3.2 Locked Test Evaluation (2014–2015)
Scored under exclusive `artifacts/TEST_LOCK` creation:

| Model Configuration | Status | Test MAE (t/ha) | Test RMSE (t/ha) | Test R² | Test MedAE (t/ha) |
|---|---|---|---|---|---|
| **`ridge_trend`** | **Selected Winner** | **0.4497** | **0.6040** | **0.5068** | **0.3444** |
| `median` | Predeclared Baseline | 0.7638 | 0.9675 | -0.2653 | 0.6406 |

### 3.3 Descriptive Year Robustness (Post-Test Breakdown)
*Note: Descriptive stability evidence across 2 test years; explicitly not an inferential confidence interval.*

| Test Year | Sample Count | MAE (t/ha) | RMSE (t/ha) | R² | MedAE (t/ha) |
|---|---|---|---|---|---|
| **2014** | 767 | 0.4415 | 0.5960 | 0.4774 | 0.3341 |
| **2015** | 79 | 0.5290 | 0.6771 | 0.4409 | 0.4514 |

### 3.4 Extension 1: Multi-Class Crop Recommendation
- **Winner:** `RandomForestClassifier`
- **Locked Test Metrics:** Accuracy = **0.9977**, Macro-F1 = **0.9977**, Precision = **0.9978**, Recall = **0.9977** (22 classes).

### 3.5 Extension 2: Depth Ablation
- Baseline Forest (`max_depth=12`): Validation MAE = **0.5761 t/ha**
- Ablated Forest (`max_depth=6`): Validation MAE = **0.6636 t/ha**
- Delta: **+0.0875 t/ha** (constraining depth increases error).

---

## 4. Five-Case Test Error Audit

Audited from the top 5 largest absolute errors on the locked test set:

| Rank | District (State, Season, Year) | Actual | Predicted | Abs Error | Agronomic Hypothesis & Agricultural Consequence | Proposed Mitigation |
|---|---|---|---|---|---|---|
| **#1** | BIJAPUR (Karnataka, Kharif 2014) | 0.05 t/ha | 2.50 t/ha | **2.45 t/ha** | **Hypothesis:** Severe sub-seasonal drought shock.<br/>**Consequence:** Deficit in local emergency grain reserve planning. | Integrate satellite NDVI and rainfall anomaly telemetry. |
| **#2** | CHIKMAGALUR (Karnataka, Kharif 2014) | 0.27 t/ha | 2.45 t/ha | **2.18 t/ha** | **Hypothesis:** Localized blast disease/lodging.<br/>**Consequence:** Procurement logistics misallocation. | Ingest pest surveillance telemetry into pre-harvest adjustments. |
| **#3** | MANDYA (Karnataka, Kharif 2014) | 0.66 t/ha | 2.82 t/ha | **2.16 t/ha** | **Hypothesis:** Lowland irrigation canal supply failure.<br/>**Consequence:** Farmer credit default risks unhedged. | Ingest canal reservoir level telemetry at block resolution. |
| **#4** | SHIMOGA (Karnataka, Summer 2014) | 0.54 t/ha | 2.46 t/ha | **1.92 t/ha** | **Hypothesis:** High temperature spike during grain filling.<br/>**Consequence:** Ineffective market procurement timing. | Fuse thermal infrared MODIS canopy temperature anomalies. |
| **#5** | MYSORE (Karnataka, Kharif 2014) | 0.77 t/ha | 2.65 t/ha | **1.88 t/ha** | **Hypothesis:** Unfavorable rainfall distribution during flowering.<br/>**Consequence:** Regional buffer storage misallocation. | Refine spatial boundary to mandal / taluk resolution. |

---

## 5. File & Artifact Structure

```
adv_pred_da8/
├── 23MID0037_Lab08.ipynb                 # Fully executed Jupyter Notebook
├── 23MID0037_Lab08_Report.pdf            # 4-page publication-quality ReportLab PDF
├── 23MID0037_Lab08_Validation_Results.csv # Validation performance comparison table
├── 23MID0037_Lab08_Test_Results.csv       # Locked test evaluation table
├── 23MID0037_Lab08_Error_Analysis.csv     # 5-case test error audit table
├── 23MID0037_Lab08_README.md             # This comprehensive README
├── lab08.py                              # Core reference implementation (Appendix A)
├── check_code.py                         # Synthetic fixture verification suite (Appendix D)
├── generate_figures.py                   # 300 DPI publication figure generator
├── generate_report.py                    # ReportLab PDF compiler
├── data/
│   └── raw/
│       ├── crop_production.csv           # Authentic GoI APY dataset (SHA-256: 5b263705...)
│       └── crop_recommendation.csv       # Extension 1 dataset (SHA-256: 808e1d84...)
├── artifacts/
│   ├── config.json                       # Experiment configuration
│   ├── versions.json                     # Environment package versions
│   ├── source_mapping.json               # Input schema and cleaning mapping
│   ├── governance_record.json            # Complete provenance & D5 audit record
│   ├── split_manifest.csv                # Row-level split mapping
│   ├── selection.json                    # Model selection record & hash seals
│   ├── acceptance.json                   # Validation assertion checks
│   ├── rolling_origins.csv               # 3 rolling origins stability evaluations
│   ├── year_robustness.csv               # Post-test year breakdown
│   ├── test_predictions.csv              # Locked test row predictions & residuals
│   ├── per_class.json                    # Classification extension metrics
│   ├── code_verification.json            # Synthetic check suite results
│   └── TEST_LOCK                         # Exclusive one-time test lockfile
├── artifacts_ablation/
│   └── ablation_comparison.json          # Isolated Extension 2 depth ablation results
├── models/
│   ├── selected_bundle.joblib            # Serialized regression winner bundle
│   └── classification_bundle.joblib      # Serialized classification winner bundle
└── figures/
    ├── fig1_training_yield_distribution.png
    ├── fig2_training_year_coverage.png
    ├── fig3_validation_model_comparison.png
    ├── fig4_locked_test_actual_vs_predicted.png
    ├── fig5_locked_test_residual_plot.png
    ├── fig6_cls_training_label_distribution.png
    ├── fig7_cls_soil_ph_distribution.png
    └── fig8_cls_confusion_matrix.png
```

---

## 6. Execution & Verification Instructions

To reproduce all results, verify assertions, and regenerate deliverables:

```bash
# 1. Run synthetic code verification suite (Appendix D)
python check_code.py

# 2. Run core data pipeline and model validation
python lab08.py --mode all

# 3. Generate high-resolution figures
python generate_figures.py

# 4. Compile publication-quality PDF report
python generate_report.py
```
