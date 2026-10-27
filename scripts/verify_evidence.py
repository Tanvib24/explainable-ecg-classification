"""Check saved predictions, selection records, and aggregate measurements."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,f1_score,average_precision_score
from ecg_project.data import ROOT,load_data,sha256,save_json
from ecg_project.experiment import source_fingerprints

results=ROOT/'results'
frozen=json.loads((results/'frozen-selection.json').read_text())
manifest=json.loads((results/'training-manifest.json').read_text())
access=json.loads((results/'test-access-log.json').read_text())
complete=json.loads((results/'test-evaluation-complete.json').read_text())
aggregate=json.loads((results/'aggregate-results.json').read_text())
x,y,meta,masks,stats=load_data()
assert frozen['source_fingerprints']==manifest['source_fingerprints']==source_fingerprints()
assert frozen['test_performance_accessed'] is False
assert frozen['frozen_utc']<=access['first_test_inference_utc']<=complete['completed_utc']
assert access['frozen_selection_sha256']==complete['frozen_selection_sha256']==sha256(results/'frozen-selection.json')
assert complete['all_runs_csv_sha256']==sha256(results/'all-runs.csv')
assert manifest['data_audit_sha256']==sha256(ROOT/'data/prepared/data-audit.json')
assert len(frozen['runs'])==12
verified=[];groups={}
for run in frozen['runs']:
    folder=results/'runs'/run['tag']
    history=pd.read_csv(folder/'history.csv')
    saved=np.load(folder/'test-predictions.npz')
    measured=json.loads((folder/'test-results.json').read_text())
    assert sha256(folder/'best.pt')==run['checkpoint_sha256']
    assert sha256(folder/'test-predictions.npz')==measured['test_predictions_sha256']
    assert np.array_equal(saved['y_true'],y[masks['test']])
    assert np.array_equal(saved['ecg_id'],meta.loc[masks['test'],'ecg_id'].to_numpy())
    assert np.array_equal(saved['patient_id'],meta.loc[masks['test'],'patient_id'].to_numpy())
    assert len(saved['y_true'])==2158 and np.isfinite(saved['y_probability']).all()
    assert np.all((saved['y_probability']>=0)&(saved['y_probability']<=1))
    assert np.allclose(saved['thresholds'],run['thresholds'])
    assert int(history.loc[history.validation_macro_auroc.idxmax(),'epoch'])==run['best_epoch']
    assert np.isclose(history.validation_macro_auroc.max(),run['validation']['macro_auroc'],atol=1e-10)
    truth=saved['y_true'];prob=saved['y_probability'];binary=prob>=saved['thresholds']
    recalculated={'macro_auroc':float(roc_auc_score(truth,prob,average='macro')),
      'macro_average_precision':float(average_precision_score(truth,prob,average='macro')),
      'macro_f1':float(f1_score(truth,binary,average='macro',zero_division=0)),
      'micro_f1':float(f1_score(truth,binary,average='micro',zero_division=0))}
    for key,value in recalculated.items():assert np.isclose(value,measured['tuned_threshold_metrics'][key],atol=1e-12)
    group=run['model']+('' if run['weighted'] else '_unweighted')
    groups.setdefault(group,[]).append(recalculated)
    verified.append(run['tag'])
for name,group in groups.items():
    assert len(group)==3
    for metric in group[0]:
        values=[row[metric] for row in group]
        assert np.isclose(np.mean(values),aggregate[name]['metrics'][metric]['mean'],atol=1e-12)
        assert np.isclose(np.std(values,ddof=1),aggregate[name]['metrics'][metric]['sample_std'],atol=1e-12)
summary={'verified_neural_runs':verified,'source_hashes_match_locked_manifest':True,
  'checkpoints_match_frozen_hashes':True,'test_ids_and_labels_match_official_fold':True,
  'saved_test_metrics_recomputed':True,'aggregate_means_and_sample_std_recomputed':True,
  'selection_frozen_before_test_access':True,'verification_uses_saved_predictions_only':True,
  'test_records':2158,'test_patients':1877}
save_json(results/'evidence-verification.json',summary)
print(json.dumps(summary,indent=2))
