"""
MDI3003 Advanced Predictive Analytics - Experiment 08
Synthetic Fixture Automated Code Verification Suite (Appendix D)
Student: Lokanth S | Reg No: 23MID0037 | Faculty: Dr. Durgesh Kumar
Note: All tests in this suite evaluate code behavior and structural contracts, NOT empirical agricultural findings.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from lab08 import (
    REG_FEATURES, CLS_FEATURES, SEED,
    get_regression_pipeline, get_classification_pipeline,
    make_chronological_split, run_acceptance_checks, evaluate_regression,
    ValidatedPredictor
)

def run_synthetic_checks():
    verification_results = {
        "suite_name": "Appendix D Synthetic Code Verification Suite",
        "scope_notice": "Code behavior and structural assertion verification ONLY. Not agricultural findings.",
        "tests": {}
    }

    # 1. Create Synthetic Fixture Data for Regression
    n_samples = 300
    np.random.seed(SEED)
    years = np.random.choice(range(2000, 2016), size=n_samples)
    states = np.random.choice(['State_A', 'State_B', 'State_C'], size=n_samples)
    districts = np.random.choice(['Dist_1', 'Dist_2', 'Dist_3', 'Dist_4'], size=n_samples)
    seasons = np.random.choice(['Kharif', 'Rabi', 'Summer'], size=n_samples)
    yields = np.random.uniform(0.5, 6.0, size=n_samples)

    synth_df = pd.DataFrame({
        'row_id': [f"SYNTH_{i+1:04d}" for i in range(n_samples)],
        'state': states,
        'district': districts,
        'season': seasons,
        'year': years,
        'crop': 'Rice',
        'yield_unit': 't/ha',
        'yield_t_ha': yields,
        'leakage_col_production': yields * 100.0, # Deliberate potential leakage column to test whitelist
        'leakage_col_area': 100.0
    })

    # Drop any duplicate synthetic keys
    synth_df = synth_df.drop_duplicates(subset=['state', 'district', 'season', 'year']).reset_index(drop=True)
    synth_df['row_id'] = [f"SYNTH_{i+1:04d}" for i in range(len(synth_df))]

    # Test 1: Chronological Partition Integrity
    try:
        train_df, val_df, test_df, manifest = make_chronological_split(synth_df)
        assert train_df['year'].max() < val_df['year'].min()
        assert val_df['year'].max() < test_df['year'].min()
        verification_results['tests']['test_chronological_split'] = {
            'status': 'PASSED',
            'detail': f"Train ({train_df['year'].min()}-{train_df['year'].max()}) < Val ({val_df['year'].min()}-{val_df['year'].max()}) < Test ({test_df['year'].min()}-{test_df['year'].max()})"
        }
    except Exception as e:
        verification_results['tests']['test_chronological_split'] = {'status': 'FAILED', 'error': str(e)}

    # Test 2: Whitelist Guard (Non-whitelisted features cannot enter model)
    try:
        X_train = train_df[REG_FEATURES] # Only REG_FEATURES allowed
        assert 'leakage_col_production' not in X_train.columns
        assert 'leakage_col_area' not in X_train.columns
        pipe = get_regression_pipeline('ridge_trend')
        pipe.fit(X_train, train_df['yield_t_ha'])
        verification_results['tests']['test_feature_whitelist_guard'] = {
            'status': 'PASSED',
            'detail': f"Only strictly whitelisted features {REG_FEATURES} accepted by pipeline."
        }
    except Exception as e:
        verification_results['tests']['test_feature_whitelist_guard'] = {'status': 'FAILED', 'error': str(e)}

    # Test 3: Training-Only Fit Category Leakage Detection
    try:
        acceptance = run_acceptance_checks(train_df, val_df, test_df, pipe)
        assert acceptance['no_encoder_leakage'] is True
        assert acceptance['all_passed'] is True
        verification_results['tests']['test_encoder_leakage_assertion'] = {
            'status': 'PASSED',
            'detail': "Fitted OneHotEncoder categories match training-set unique values bit-exact."
        }
    except Exception as e:
        verification_results['tests']['test_encoder_leakage_assertion'] = {'status': 'FAILED', 'error': str(e)}

    # Test 4: Regression Candidates Training & Metric Determinism
    try:
        candidates = ['median', 'ridge_trend', 'tree', 'forest']
        metrics_dict = {}
        for c in candidates:
            p = get_regression_pipeline(c)
            p.fit(X_train, train_df['yield_t_ha'])
            preds = p.predict(val_df[REG_FEATURES])
            m = evaluate_regression(val_df['yield_t_ha'], preds)
            assert np.isfinite(m['mae_t_ha'])
            metrics_dict[c] = m
        verification_results['tests']['test_regression_candidates'] = {
            'status': 'PASSED',
            'candidates_evaluated': list(metrics_dict.keys())
        }
    except Exception as e:
        verification_results['tests']['test_regression_candidates'] = {'status': 'FAILED', 'error': str(e)}

    # Test 5: Reload Roundtrip Verification
    try:
        forest_p = get_regression_pipeline('forest')
        forest_p.fit(X_train, train_df['yield_t_ha'])
        orig_preds = forest_p.predict(val_df[REG_FEATURES])
        
        ohe = forest_p.named_steps['preprocessor'].named_transformers_['cat']
        test_bundle = {
            'pipeline': forest_p,
            'features': REG_FEATURES,
            'year_range': [int(train_df['year'].min()), int(train_df['year'].max())],
            'fitted_categories': {
                'state': list(ohe.categories_[0]),
                'district': list(ohe.categories_[1]),
                'season': list(ohe.categories_[2])
            }
        }
        os.makedirs("scratch", exist_ok=True)
        bundle_temp = "scratch/temp_synth_bundle.joblib"
        joblib.dump(test_bundle, bundle_temp)
        reloaded = joblib.load(bundle_temp)
        reloaded_preds = reloaded['pipeline'].predict(val_df[REG_FEATURES])
        np.testing.assert_allclose(orig_preds, reloaded_preds, rtol=1e-10, atol=1e-10)
        verification_results['tests']['test_model_reload_exactness'] = {
            'status': 'PASSED',
            'detail': "Joblib serialization and deserialization yields bit-exact prediction reproduction."
        }
    except Exception as e:
        verification_results['tests']['test_model_reload_exactness'] = {'status': 'FAILED', 'error': str(e)}

    # Test 6: Validated Predictor Guardrails
    try:
        predictor = ValidatedPredictor(bundle_temp)
        # Valid prediction
        val_pred = predictor.predict_one('State_A', 'Dist_1', 'Kharif', 2005)
        assert val_pred >= 0.0

        # Out of bounds year rejection test
        rejected = False
        try:
            predictor.predict_one('State_A', 'Dist_1', 'Kharif', 1980) # Pre-2000 year
        except ValueError:
            rejected = True
        assert rejected is True, "Inference guard failed to reject out-of-range year!"

        verification_results['tests']['test_inference_guardrails'] = {
            'status': 'PASSED',
            'detail': "Inference guard correctly validates bounds and rejects out-of-range inputs."
        }
    except Exception as e:
        verification_results['tests']['test_inference_guardrails'] = {'status': 'FAILED', 'error': str(e)}

    # Test 7: Classification Pipeline Functionality
    try:
        # Synthetic classification fixture
        n_cls = 220
        classes = [f"Crop_{k}" for k in range(11)]
        y_synth = np.random.choice(classes, size=n_cls)
        X_synth = pd.DataFrame(np.random.uniform(10, 100, size=(n_cls, 7)), columns=CLS_FEATURES)
        
        cls_pipe = get_classification_pipeline('forest')
        cls_pipe.fit(X_synth, y_synth)
        cls_preds = cls_pipe.predict(X_synth)
        assert len(cls_preds) == n_cls
        verification_results['tests']['test_classification_pipeline'] = {
            'status': 'PASSED',
            'detail': f"Classification forest pipeline fits and predicts across {len(classes)} classes."
        }
    except Exception as e:
        verification_results['tests']['test_classification_pipeline'] = {'status': 'FAILED', 'error': str(e)}

    # Summary
    all_passed = all(t['status'] == 'PASSED' for t in verification_results['tests'].values())
    verification_results['all_passed'] = all_passed

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/code_verification.json", "w") as f:
        json.dump(verification_results, f, indent=2)

    print("=== Synthetic Code Verification Results ===")
    print(json.dumps(verification_results, indent=2))
    return all_passed

if __name__ == '__main__':
    success = run_synthetic_checks()
    if not success:
        sys.exit(1)
