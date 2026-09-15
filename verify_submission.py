"""
Automated Cross-Artifact Consistency Verification Script for Lab 08.
Checks that all required files, hashes, and metric numbers agree across all artifacts.
"""

import os
import json
import hashlib
import pandas as pd
import pypdf

def check_cross_artifacts():
    print("=== Cross-Artifact Consistency Verification ===")
    
    # 1. Check required student deliverables
    required_root_files = [
        "23MID0037_Lab08.ipynb",
        "23MID0037_Lab08_Validation_Results.csv",
        "23MID0037_Lab08_Classification_Validation_Results.csv",
        "23MID0037_Lab08_Test_Results.csv",
        "23MID0037_Lab08_Error_Analysis.csv",
        "23MID0037_Lab08_Report.pdf",
        "23MID0037_Lab08_Report.docx",
        "23MID0037_Lab08_README.md",
        "lab08.py",
        "check_code.py",
        "generate_figures.py"
    ]
    for rf in required_root_files:
        assert os.path.exists(rf), f"Missing root file: {rf}"
        print(f"[OK] Root file exists: {rf}")

    # Check report page count (target: ~4-6 pages)
    pdf_reader = pypdf.PdfReader("23MID0037_Lab08_Report.pdf")
    num_pages = len(pdf_reader.pages)
    assert 4 <= num_pages <= 7, f"Report page count out of bounds (~4-6 pages expected, found {num_pages})"
    print(f"[OK] Report PDF verified: {num_pages} pages (target ~4-6 pages)")

    # 2. Check required folders
    for d in ["models", "figures", "artifacts", "artifacts_ablation", "data/raw"]:
        assert os.path.isdir(d), f"Missing directory: {d}"
        print(f"[OK] Directory exists: {d}")

    # 3. Check artifacts files
    required_artifacts = [
        "artifacts/config.json",
        "artifacts/versions.json",
        "artifacts/source_mapping.json",
        "artifacts/governance_record.json",
        "artifacts/split_manifest.csv",
        "artifacts/selection.json",
        "artifacts/acceptance.json",
        "artifacts/rolling_origins.csv",
        "artifacts/year_robustness.csv",
        "artifacts/test_predictions.csv",
        "artifacts/per_class.json",
        "artifacts/code_verification.json",
        "artifacts/d5_probe_log.json",
        "artifacts/TEST_LOCK",
        "artifacts_ablation/ablation_comparison.json",
        "models/selected_bundle.joblib",
        "models/classification_bundle.joblib"
    ]
    for af in required_artifacts:
        assert os.path.exists(af), f"Missing artifact: {af}"
        print(f"[OK] Artifact exists: {af}")

    # 4. Check figures count
    figs = [f"figures/fig{i}_{suffix}.png" for i, suffix in [
        (1, "training_yield_distribution"),
        (2, "training_year_coverage"),
        (3, "validation_model_comparison"),
        (4, "locked_test_actual_vs_predicted"),
        (5, "locked_test_residual_plot"),
        (6, "cls_training_label_distribution"),
        (7, "cls_soil_ph_distribution"),
        (8, "cls_confusion_matrix")
    ]]
    for fig in figs:
        assert os.path.exists(fig), f"Missing figure: {fig}"
        assert os.path.getsize(fig) > 10000, f"Figure too small / empty: {fig}"
        print(f"[OK] Figure verified: {fig} ({os.path.getsize(fig)} bytes)")

    # 5. Check real data SHA-256
    with open("data/raw/crop_production.csv", "rb") as f:
        real_d1_hash = hashlib.sha256(f.read()).hexdigest()
    with open("artifacts/selection.json", "r") as f:
        sel = json.load(f)
    with open("artifacts/governance_record.json", "r") as f:
        gov = json.load(f)
    
    assert real_d1_hash == sel['data_sha256'], "Data hash mismatch in selection.json!"
    assert real_d1_hash == gov['d1_ingestion_record']['sha256'], "Data hash mismatch in governance_record.json!"
    print(f"[OK] Data SHA-256 matches perfectly: {real_d1_hash}")

    # 6. Check validation table vs selection.json
    val_csv = pd.read_csv("23MID0037_Lab08_Validation_Results.csv")
    winner_csv = val_csv.iloc[0]
    assert winner_csv['model'] == sel['selected_model']
    assert abs(winner_csv['val_mae_t_ha'] - sel['val_mae_t_ha']) < 1e-4
    print(f"[OK] Validation winner verified: {sel['selected_model']} (Val MAE: {sel['val_mae_t_ha']} t/ha)")

    # 7. Check acceptance.json
    with open("artifacts/acceptance.json", "r") as f:
        acc = json.load(f)
    assert acc['all_passed'] is True
    print(f"[OK] Acceptance assertions all passed: {acc['all_passed']}")

    # 8. Check code_verification.json
    with open("artifacts/code_verification.json", "r") as f:
        cv = json.load(f)
    assert cv['all_passed'] is True
    print(f"[OK] Synthetic code verification suite all passed: {cv['all_passed']}")

    # 9. Check notebook execution state
    with open("23MID0037_Lab08.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    for i, c in enumerate(code_cells):
        assert len(c.get('outputs', [])) > 0, f"Code cell {i+1} has no executed outputs!"
    print(f"[OK] Notebook fully executed: all {len(code_cells)} code cells contain rich outputs!")

    print("\n=== ALL CROSS-ARTIFACT CONSISTENCY CHECKS PASSED WITH ZERO ERRORS ===")

if __name__ == '__main__':
    check_cross_artifacts()
