from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .data import ROOT, CACHE, CLASSES, load_data, save_json, sha256
from .models import build_model, parameter_count

RESULTS=ROOT/'results'
PROTOCOL=ROOT/'configs'/'protocol.json'


def metrics(y,p,thresholds=0.5):
    binary=p>=thresholds
    auc=roc_auc_score(y,p,average=None)
    ap=average_precision_score(y,p,average=None)
    f1=f1_score(y,binary,average=None,zero_division=0)
    precision=precision_score(y,binary,average=None,zero_division=0)
    recall=recall_score(y,binary,average=None,zero_division=0)
    return {'macro_auroc':float(auc.mean()),'macro_average_precision':float(ap.mean()),'macro_f1':float(f1.mean()),
            'micro_f1':float(f1_score(y,binary,average='micro',zero_division=0)),
            'per_class':{name:{'auroc':float(auc[c]),'average_precision':float(ap[c]),'f1':float(f1[c]),
               'precision':float(precision[c]),'sensitivity':float(recall[c]),'support':int(y[:,c].sum()),
               'prevalence':float(y[:,c].mean())} for c,name in enumerate(CLASSES)}}


def thresholds_from_validation(y,p):
    grid=np.round(np.arange(0.05,0.951,0.01),2)
    selected=[]
    for c in range(5):
        scores=np.array([f1_score(y[:,c],p[:,c]>=t,zero_division=0) for t in grid])
        selected.append(float(grid[np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-12))[-1]]))
    return np.asarray(selected,dtype=np.float32)


def configure_torch(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def normalized_arrays():
    x,y,meta,masks,stats=load_data()
    normalized=np.array(x,dtype=np.float32,copy=True)
    normalized-=stats['mean'][None,:,None]
    normalized/=(stats['std'][None,:,None]+1e-6)
    if not np.isfinite(normalized).all():raise ValueError('Invalid normalized data')
    return torch.from_numpy(normalized),y,meta,masks,stats


@torch.inference_mode()
def predict(model,x,batch_size=128):
    model.eval()
    outputs=[]
    for start in range(0,len(x),batch_size):outputs.append(torch.sigmoid(model(x[start:start+batch_size])).numpy())
    return np.concatenate(outputs)


def source_fingerprints():
    return {str(p.relative_to(ROOT)):sha256(p) for p in [PROTOCOL,ROOT/'ecg_project'/'models.py',ROOT/'ecg_project'/'data.py',Path(__file__),ROOT/'requirements-lock.txt']}


def environment_record():
    result={'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
            'torch_version':torch.__version__,'cuda_available':torch.cuda.is_available(),'device':'cpu',
            'logical_cpu_count':os.cpu_count(),'packages':{name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','scikit-learn','scipy','wfdb','matplotlib']},
            'torch_threads':torch.get_num_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    return result


def train_one(name,seed,weighted,config,train_x,train_y,val_x,val_y,weights):
    tag=f'{name}_seed{seed}'+('' if weighted else '_unweighted')
    folder=RESULTS/'runs'/tag;folder.mkdir(parents=True,exist_ok=True)
    best_path=folder/'best.pt';last_path=folder/'last.pt';complete=folder/'training-summary.json'
    if complete.exists():
        result=json.loads(complete.read_text())
        if result['source_fingerprints']!=source_fingerprints():raise RuntimeError('Completed run uses different source/configuration')
        print('Already completed: '+tag,flush=True);return result
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    generator=torch.Generator().manual_seed(seed)
    model=build_model(name,config['dropout'],config['model_profile'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(weights) if weighted else None)
    history=[];best=-float('inf');stop_best=best;stale=0;start_epoch=0;accumulated=0.0;best_epoch=0
    if last_path.exists():
        state=torch.load(last_path,map_location='cpu',weights_only=True)
        if state['source_fingerprints']!=source_fingerprints():raise RuntimeError('Cannot resume after source/configuration drift')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);generator.set_state(state['shuffle_rng'])
        history=state['history'];best=state['best'];stop_best=state['stop_best'];stale=state['stale'];start_epoch=state['epoch'];accumulated=state['seconds'];best_epoch=state['best_epoch']
    started=time.perf_counter()
    if stale<config['early_stopping_patience']:
        for epoch in range(start_epoch,config['max_epochs']):
            epoch_started=time.perf_counter();model.train();total_loss=0.0
            order=torch.randperm(len(train_x),generator=generator)
            for start in range(0,len(order),config['batch_size']):
                indices=order[start:start+config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                logits=model(train_x[indices])
                loss=criterion(logits,train_y[indices])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss in '+tag)
                loss.backward();optimizer.step();total_loss+=float(loss.detach())*len(indices)
            probabilities=predict(model,val_x)
            val_auc=float(roc_auc_score(val_y,probabilities,average='macro'))
            val_ap=float(average_precision_score(val_y,probabilities,average='macro'))
            row={'epoch':epoch+1,'training_loss':total_loss/len(train_x),'validation_macro_auroc':val_auc,
                 'validation_macro_average_precision':val_ap,'seconds':time.perf_counter()-epoch_started}
            history.append(row)
            if val_auc>best:
                best=val_auc;best_epoch=epoch+1
                torch.save({'model':model.state_dict(),'name':name,'seed':seed,'weighted':weighted,'epoch':best_epoch,'validation_macro_auroc':best,'config':config},best_path)
            if val_auc>stop_best+config['minimum_improvement']:stop_best=val_auc;stale=0
            else:stale+=1
            elapsed=accumulated+time.perf_counter()-started
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                   'shuffle_rng':generator.get_state(),'history':history,'best':best,'best_epoch':best_epoch,
                   'stop_best':stop_best,'stale':stale,'epoch':epoch+1,'seconds':elapsed,'source_fingerprints':source_fingerprints()}
            temporary=last_path.with_suffix('.partial');torch.save(state,temporary);temporary.replace(last_path)
            pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
            save_json(folder/'status.json',{'run':tag,'epoch':epoch+1,'max_epochs':config['max_epochs'],'best_validation_macro_auroc':best,'run_seconds':elapsed,'phase':'training'})
            print(f'{tag}: epoch {epoch+1}/{config["max_epochs"]}; loss={row["training_loss"]:.4f}; val AUROC={val_auc:.4f}; {row["seconds"]:.1f}s',flush=True)
            if stale>=config['early_stopping_patience']:break
    state=torch.load(best_path,map_location='cpu',weights_only=True);model.load_state_dict(state['model'])
    val_p=predict(model,val_x);thresholds=thresholds_from_validation(val_y,val_p)
    np.savez(folder/'validation-predictions.npz',y_true=val_y,y_probability=val_p,thresholds=thresholds)
    summary={'tag':tag,'model':name,'seed':seed,'weighted':weighted,'parameters':parameter_count(model),
             'epochs_run':len(history),'best_epoch':best_epoch,'training_seconds':accumulated+time.perf_counter()-started,
             'thresholds':thresholds.tolist(),'validation':metrics(val_y,val_p,thresholds),
             'source_fingerprints':source_fingerprints(),'checkpoint_sha256':sha256(best_path),
             'test_evaluated':False}
    save_json(complete,summary)
    return summary


def engineered_features(x):
    """Eight non-diagnostic waveform summary statistics per lead (96 features)."""
    features=[]
    for start in range(0,len(x),256):
        values=x[start:start+256].numpy();diff=np.diff(values,axis=2)
        features.append(np.concatenate([values.mean(2),values.std(2),values.min(2),values.max(2),
               np.sqrt(np.mean(values**2,axis=2)),np.abs(diff).mean(2),np.quantile(values,0.05,axis=2),np.quantile(values,0.95,axis=2)],axis=1).astype('float32'))
    return np.concatenate(features)


def train_all():
    config=json.loads(PROTOCOL.read_text())
    RESULTS.mkdir(exist_ok=True)
    if (RESULTS/'test-evaluation-complete.json').exists():raise RuntimeError('Test evaluation already complete; do not retune this experiment')
    configure_torch(config['torch_threads'])
    fingerprints=source_fingerprints()
    if (RESULTS/'training-manifest.json').exists():
        previous=json.loads((RESULTS/'training-manifest.json').read_text())
        if previous['source_fingerprints']!=fingerprints:raise RuntimeError('Training manifest is already locked to other code/configuration')
    else:
        save_json(RESULTS/'training-manifest.json',{'source_fingerprints':fingerprints,'protocol':config,
                   'environment':environment_record(),'data_audit_sha256':sha256(CACHE/'data-audit.json'),
                   'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    jobs=[(name,seed,True) for name in config['models'] for seed in config['seeds']]
    jobs.extend([('compact_cnn',seed,False) for seed in config['seeds']])
    def launch(job):
        name,seed,weighted=job
        tag=f'{name}_seed{seed}'+('' if weighted else '_unweighted')
        folder=RESULTS/'runs'/tag;folder.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-m','ecg_project.experiment','train-run','--model',name,'--seed',str(seed)]
        if not weighted:command.append('--unweighted')
        env=dict(os.environ,OMP_NUM_THREADS=str(config['torch_threads']),MKL_NUM_THREADS=str(config['torch_threads']))
        with (folder/'training.log').open('a',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed '+tag,flush=True)
        return json.loads((folder/'training-summary.json').read_text())
    save_json(RESULTS/'training-status.json',{'phase':'training','jobs':len(jobs),'parallel_processes':config['parallel_training_processes']})
    with ThreadPoolExecutor(max_workers=config['parallel_training_processes']) as pool:
        runs=list(pool.map(launch,jobs))
    x,y,meta,masks,stats=normalized_arrays()
    val_y=y[masks['validation']]
    # Feature baseline selection uses only train and validation partitions.
    features=engineered_features(x)
    np.save(CACHE/'summary-features.npy',features)
    best=-1;choices=[];selected=None
    for c in [0.1,1.0,10.0]:
        baseline=make_pipeline(StandardScaler(),OneVsRestClassifier(LogisticRegression(C=c,class_weight='balanced',max_iter=1000)))
        start=time.perf_counter();baseline.fit(features[masks['train']],y[masks['train']])
        val_p=baseline.predict_proba(features[masks['validation']])
        auc=float(roc_auc_score(val_y,val_p,average='macro'))
        choices.append({'C':c,'validation_macro_auroc':auc,'fit_seconds':time.perf_counter()-start})
        if auc>best:best=auc;selected=(baseline,c,val_p)
    baseline,c,val_p=selected
    joblib.dump(baseline,RESULTS/'logistic-baseline.joblib')
    thresholds=thresholds_from_validation(val_y,val_p)
    np.savez(RESULTS/'logistic-validation-predictions.npz',y_true=val_y,y_probability=val_p,thresholds=thresholds)
    save_json(RESULTS/'logistic-selection.json',{'C':c,'candidates':choices,'thresholds':thresholds.tolist(),
        'validation':metrics(val_y,val_p,thresholds),'feature_description':'8 summary statistics per lead, 96 total; preprocessing fitted only to training data',
        'convergence_iterations':[int(estimator.n_iter_[0]) for estimator in baseline[-1].estimators_],
        'model_sha256':sha256(RESULTS/'logistic-baseline.joblib')})
    winner=max((run for run in runs if run['weighted'] and run['seed']==42),key=lambda r:r['validation']['macro_auroc'])['tag']
    save_json(RESULTS/'frozen-selection.json',{'runs':runs,'prespecified_explanation_checkpoint':winner,
        'selection_rule':'highest validation macro AUROC among weighted seed-42 checkpoints',
        'source_fingerprints':fingerprints,'test_performance_accessed':False,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    save_json(RESULTS/'training-status.json',{'phase':'all_checkpoints_and_thresholds_frozen','runs':len(runs)})
    print('Training complete. Checkpoints and thresholds frozen before test evaluation.',flush=True)


def latency(model,x,config):
    model.eval();times=[]
    with torch.inference_mode():
        for i in range(config['latency']['warmup']):model(x[i%len(x):i%len(x)+1])
        for i in range(config['latency']['repeats']):
            sample=x[i%len(x):i%len(x)+1]
            start=time.perf_counter_ns();model(sample);times.append((time.perf_counter_ns()-start)/1e6)
    return {'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,0.95)),'measurements_ms':times,
            'threads':torch.get_num_threads(),'batch_size':1,'scope':'model forward only'}


def bootstrap_intervals(y,p,thresholds,patients,repeats=1000,seed=1997):
    unique,inverse=np.unique(patients,return_inverse=True)
    groups=[np.flatnonzero(inverse==i) for i in range(len(unique))]
    rng=np.random.default_rng(seed);samples=[];undefined=0
    for b in range(repeats):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        yy=y[indices];pp=p[indices]
        if np.any(yy.sum(0)==0) or np.any(yy.sum(0)==len(yy)):
            undefined+=1;continue
        mm=metrics(yy,pp,thresholds)
        samples.append([mm['macro_auroc'],mm['macro_average_precision'],mm['macro_f1'],mm['micro_f1']])
    values=np.asarray(samples)
    return {'unit':'patient','replicates_requested':repeats,'replicates_valid':len(values),'undefined_replicates':undefined,
       'seed':seed,'confidence_level':0.95,'intervals':{name:np.quantile(values[:,i],[0.025,0.975]).tolist()
          for i,name in enumerate(['macro_auroc','macro_average_precision','macro_f1','micro_f1'])}}


def evaluate_all():
    frozen=json.loads((RESULTS/'frozen-selection.json').read_text())
    config=json.loads(PROTOCOL.read_text())
    if frozen['source_fingerprints']!=source_fingerprints():raise RuntimeError('Source changed after freezing')
    if (RESULTS/'test-evaluation-complete.json').exists():
        print('Evaluation already completed; use saved predictions and measured results.',flush=True);return
    configure_torch(config['torch_threads'])
    save_json(RESULTS/'test-access-log.json',{'frozen_selection_sha256':sha256(RESULTS/'frozen-selection.json'),
        'first_test_inference_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    x,y,meta,masks,stats=normalized_arrays()
    test_x=x[np.flatnonzero(masks['test'])];test_y=y[masks['test']]
    patients=meta.loc[masks['test'],'patient_id'].to_numpy();ids=meta.loc[masks['test'],'ecg_id'].to_numpy()
    rows=[]
    for run in frozen['runs']:
        folder=RESULTS/'runs'/run['tag'];summary_path=folder/'test-results.json'
        if summary_path.exists():rows.append(json.loads(summary_path.read_text()));continue
        if sha256(folder/'best.pt')!=run['checkpoint_sha256']:raise RuntimeError('Frozen checkpoint changed')
        model=build_model(run['model'],config['dropout'],config['model_profile'])
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model'])
        started=time.perf_counter();p=predict(model,test_x)
        thresholds=np.asarray(run['thresholds'])
        np.savez(folder/'test-predictions.npz',y_true=test_y,y_probability=p,thresholds=thresholds,ecg_id=ids,patient_id=patients)
        measured={'tag':run['tag'],'model':run['model'],'seed':run['seed'],'weighted':run['weighted'],
            'parameters':run['parameters'],'best_epoch':run['best_epoch'],'epochs_run':run['epochs_run'],
            'training_seconds':run['training_seconds'],'tuned_threshold_metrics':metrics(test_y,p,thresholds),
            'fixed_threshold_metrics':metrics(test_y,p,0.5),'latency':latency(model,test_x,config),
            'test_predictions_sha256':sha256(folder/'test-predictions.npz'),'test_forward_seconds':time.perf_counter()-started}
        if run['seed']==42:
            measured['bootstrap']=bootstrap_intervals(test_y,p,thresholds,patients,config['bootstrap']['replicates'],config['bootstrap']['seed'])
        save_json(summary_path,measured);rows.append(measured)
        print(f'Evaluated {run["tag"]}: AUROC={measured["tuned_threshold_metrics"]["macro_auroc"]:.4f}; macro F1={measured["tuned_threshold_metrics"]["macro_f1"]:.4f}',flush=True)
    features=np.load(CACHE/'summary-features.npy')
    baseline=joblib.load(RESULTS/'logistic-baseline.joblib');selection=json.loads((RESULTS/'logistic-selection.json').read_text())
    if sha256(RESULTS/'logistic-baseline.joblib')!=selection['model_sha256']:raise RuntimeError('Frozen logistic model changed')
    p=baseline.predict_proba(features[masks['test']]);thresholds=np.asarray(selection['thresholds'])
    np.savez(RESULTS/'logistic-test-predictions.npz',y_true=test_y,y_probability=p,thresholds=thresholds,ecg_id=ids,patient_id=patients)
    save_json(RESULTS/'logistic-test-results.json',{'tuned_threshold_metrics':metrics(test_y,p,thresholds),
       'fixed_threshold_metrics':metrics(test_y,p,0.5),'bootstrap':bootstrap_intervals(test_y,p,thresholds,patients),
       'note':'Logistic baseline uses 96 waveform summary features; latency is not directly compared to raw-signal CNN forward latency.'})
    flat=[]
    for row in rows:
        entry={k:row[k] for k in ['tag','model','seed','weighted','parameters','best_epoch','epochs_run','training_seconds']}
        entry.update({k:v for k,v in row['tuned_threshold_metrics'].items() if k!='per_class'})
        entry.update({'latency_median_ms':row['latency']['median_ms'],'latency_p95_ms':row['latency']['p95_ms']})
        flat.append(entry)
    pd.DataFrame(flat).to_csv(RESULTS/'all-runs.csv',index=False)
    summary={}
    for (name,weighted),group in pd.DataFrame(flat).groupby(['model','weighted']):
        summary[name+('' if weighted else '_unweighted')]={'seeds':group.seed.astype(int).tolist(),
          'parameters':int(group.parameters.iloc[0]),'metrics':{metric:{'mean':float(group[metric].mean()),'sample_std':float(group[metric].std(ddof=1))}
             for metric in ['macro_auroc','macro_average_precision','macro_f1','micro_f1','latency_median_ms','training_seconds']}}
    save_json(RESULTS/'aggregate-results.json',summary)
    save_json(RESULTS/'test-evaluation-complete.json',{'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'runs_evaluated':len(rows),'frozen_selection_sha256':sha256(RESULTS/'frozen-selection.json'),'test_records':len(test_y),
      'selection_used_test_metrics':False,'all_runs_csv_sha256':sha256(RESULTS/'all-runs.csv')})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['train','train-run','evaluate'])
    parser.add_argument('--model',choices=['compact_cnn','resnet1d','inception_time']);parser.add_argument('--seed',type=int);parser.add_argument('--unweighted',action='store_true')
    args=parser.parse_args()
    if args.action=='train':train_all()
    elif args.action=='evaluate':evaluate_all()
    else:
        config=json.loads(PROTOCOL.read_text());configure_torch(config['torch_threads'])
        manifest=json.loads((RESULTS/'training-manifest.json').read_text())
        if manifest['source_fingerprints']!=source_fingerprints():raise RuntimeError('Worker source differs from the locked manifest')
        x,y,meta,masks,stats=normalized_arrays()
        train_one(args.model,args.seed,not args.unweighted,config,x[np.flatnonzero(masks['train'])],torch.from_numpy(y[masks['train']]),
                  x[np.flatnonzero(masks['validation'])],y[masks['validation']],stats['pos_weight'])
