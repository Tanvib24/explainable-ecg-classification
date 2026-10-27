"""Create scientific figures and occlusion analyses from frozen experiments."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from scipy.stats import spearmanr
import torch
from ecg_project.data import ROOT,CLASSES,save_json,sha256
from ecg_project.models import build_model
from ecg_project.experiment import normalized_arrays,predict,configure_torch

RESULTS=ROOT/'results';FIG=RESULTS/'figures'
NAMES={'compact_cnn':'Compact CNN','resnet1d':'1D ResNet','inception_time':'InceptionTime variant'}
LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                     'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'savefig.dpi':300,
                     'axes.spines.top':False,'axes.spines.right':False})


def savefig(name):
    plt.tight_layout(pad=0.8);plt.savefig(FIG/(name+'.png'),bbox_inches='tight',facecolor='white');plt.close()


def performance_figures(config,frozen):
    colors={'compact_cnn':'#173f5f','resnet1d':'#aa4a10','inception_time':'#35734e'}
    fig,ax=plt.subplots(figsize=(3.45,2.45))
    for name in config['models']:
        histories=[]
        for seed in config['seeds']:
            history=pd.read_csv(RESULTS/'runs'/f'{name}_seed{seed}'/'history.csv')
            ax.plot(history.epoch,history.validation_macro_auroc,color=colors[name],alpha=.25,lw=.7)
            if seed==42:ax.plot(history.epoch,history.validation_macro_auroc,color=colors[name],lw=1.5,label=NAMES[name])
    ax.set(xlabel='Training epoch',ylabel='Validation macro AUROC',ylim=(.75,.97))
    ax.grid(alpha=.2);ax.legend(loc='lower right',frameon=False)
    savefig('validation_learning_curves')
    winner=frozen['prespecified_explanation_checkpoint']
    saved=np.load(RESULTS/'runs'/winner/'test-predictions.npz')
    y=saved['y_true'];p=saved['y_probability'];thresholds=saved['thresholds']
    fig,ax=plt.subplots(figsize=(3.45,2.55))
    for c,name in enumerate(CLASSES):
        precision,recall,_=precision_recall_curve(y[:,c],p[:,c])
        ax.plot(recall,precision,lw=1,label=name)
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02));ax.legend(frameon=False,fontsize=7)
    ax.grid(alpha=.2);savefig('test_precision_recall')
    fig,ax=plt.subplots(figsize=(3.45,2.55))
    for c,name in enumerate(CLASSES):
        fpr,tpr,_=roc_curve(y[:,c],p[:,c]);ax.plot(fpr,tpr,label=name,lw=1)
    ax.plot([0,1],[0,1],color='gray',ls='--',lw=.6)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1.02));ax.legend(frameon=False)
    savefig('test_roc')
    fig,axes=plt.subplots(2,3,figsize=(7,4.2))
    matrix_rows=[]
    for c,name in enumerate(CLASSES):
        matrix=confusion_matrix(y[:,c],p[:,c]>=thresholds[c],labels=[0,1]);ax=axes.flat[c]
        ax.imshow(matrix,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black',fontsize=10)
        ax.set(title=name,xlabel='Predicted label',ylabel='True label',xticks=[0,1],yticks=[0,1]);ax.tick_params(length=0)
        tn,fp,fn,tp=matrix.ravel();matrix_rows.append({'class':name,'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)})
    axes.flat[-1].axis('off');savefig('test_confusion_matrices')
    pd.DataFrame(matrix_rows).to_csv(RESULTS/'per-class-confusion-counts.csv',index=False)
    rows=[]
    for run in frozen['runs']:
        measured=json.loads((RESULTS/'runs'/run['tag']/'test-results.json').read_text())
        for label,values in measured['tuned_threshold_metrics']['per_class'].items():
            rows.append({'tag':run['tag'],'model':run['model'],'seed':run['seed'],'weighted':run['weighted'],'class':label,**values})
    pd.DataFrame(rows).to_csv(RESULTS/'per-class-results.csv',index=False)


def occlusion_analysis(config,frozen):
    configure_torch(config['torch_threads'])
    x,y,meta,masks,stats=normalized_arrays()
    test_x=x[np.flatnonzero(masks['test'])];test_y=y[masks['test']];test_meta=meta.loc[masks['test']].reset_index(drop=True)
    winner=frozen['prespecified_explanation_checkpoint'];folder=RESULTS/'runs'/winner
    record=next(run for run in frozen['runs'] if run['tag']==winner)
    model=build_model(record['model'],config['dropout'],config['model_profile'])
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['model'])
    saved=np.load(folder/'test-predictions.npz');p=saved['y_probability'];thresholds=saved['thresholds']
    assert np.array_equal(saved['ecg_id'],test_meta.ecg_id.to_numpy())
    cases=[];window=config['occlusion']['window_samples'];step=config['occlusion']['stride_samples'];starts=np.arange(0,1000-window+1,step)
    for c,label in enumerate(CLASSES):
        for kind,mask in [('true_positive',(test_y[:,c]==1)&(p[:,c]>=thresholds[c])),('false_negative',(test_y[:,c]==1)&(p[:,c]<thresholds[c]))]:
            eligible=np.flatnonzero(mask)
            if not len(eligible):cases.append({'class':label,'kind':kind,'available':False});continue
            i=int(eligible[np.argmin(test_meta.ecg_id.to_numpy()[eligible])])
            signal=test_x[i:i+1].clone();base=float(p[i,c]);live=float(predict(model,signal)[0,c])
            if not np.isclose(base,live,atol=2e-6):raise ValueError('Saved prediction is not reproducible for explanation')
            responses={}
            for baseline in ['zero','record_median']:
                fill=torch.zeros(12) if baseline=='zero' else signal[0].median(dim=1).values
                variants=signal.repeat(12+len(starts),1,1)
                for lead in range(12):variants[lead,lead,:]=fill[lead]
                for j,start in enumerate(starts):variants[12+j,:,start:start+window]=fill[:,None]
                scores=predict(model,variants)[:,c]
                responses[baseline]={'lead_delta':(base-scores[:12]).tolist(),'temporal_delta':(base-scores[12:]).tolist()}
            zero=responses['zero'];median=responses['record_median']
            lead_r=float(spearmanr(zero['lead_delta'],median['lead_delta']).statistic)
            temporal_r=float(spearmanr(zero['temporal_delta'],median['temporal_delta']).statistic)
            info={'class':label,'kind':kind,'available':True,'ecg_id':int(test_meta.iloc[i].ecg_id),
                  'patient_id':int(test_meta.iloc[i].patient_id),'true_labels':[CLASSES[j] for j in np.flatnonzero(test_y[i])],
                  'probability':base,'threshold':float(thresholds[c]),'prediction_reproduction_absolute_error':abs(base-live),
                  'responses':responses,'baseline_sensitivity':{'lead_spearman':lead_r if np.isfinite(lead_r) else None,
                      'temporal_spearman':temporal_r if np.isfinite(temporal_r) else None},
                  'selection':'lowest ECG identifier among eligible examples; no visual cherry-picking'}
            cases.append(info)
            case_name=f'occlusion_{label}_{kind}'
            np.savez(FIG/(case_name+'.npz'),waveform=signal.numpy(),start_samples=starts,probability=base,
                     lead_delta=zero['lead_delta'],temporal_delta=zero['temporal_delta'],alternate_temporal_delta=median['temporal_delta'])
            fig,axes=plt.subplots(2,1,figsize=(3.45,2.75),sharex=True,gridspec_kw={'height_ratios':[1,1]})
            axes[0].plot(np.arange(1000)/100,signal[0,1].numpy(),lw=.6,color='black')
            axes[0].set(ylabel='Lead II (normalized)',title=f'{label} {kind.replace("_"," ")}: ECG {info["ecg_id"]}')
            centers=(starts+window/2)/100
            axes[1].plot(centers,zero['temporal_delta'],label='Zero baseline',color='#173f5f',lw=1)
            axes[1].plot(centers,median['temporal_delta'],label='Median baseline',ls='--',color='#aa4a10',lw=.8)
            axes[1].axhline(0,color='gray',lw=.5)
            axes[1].set(xlabel='Time (s)',ylabel='Change in category score',xlim=(0,10));axes[1].legend(frameon=False,fontsize=6)
            savefig(case_name)
            fig,ax=plt.subplots(figsize=(3.45,2.1));ax.bar(LEADS,zero['lead_delta'],color='#173f5f')
            ax.axhline(0,color='gray',lw=.5);ax.set(ylabel='Change in category score',title=f'{label}: ECG {info["ecg_id"]}');ax.tick_params(axis='x',rotation=45)
            savefig(case_name+'_leads')
    save_json(RESULTS/'occlusion-analysis.json',{'checkpoint':winner,'selection_frozen_before_test':True,'window_samples':window,
        'stride_samples':step,'lead_order':LEADS,'cases':cases,
        'interpretation_limit':'Input perturbation sensitivity, not causal physiology or clinician-validated localization.'})
    print('Saved original occlusion measurements for '+str(sum(case['available'] for case in cases))+' examples',flush=True)


if __name__=='__main__':
    if not (RESULTS/'test-evaluation-complete.json').exists():raise RuntimeError('Frozen test evaluation must complete first')
    FIG.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs'/'protocol.json').read_text());frozen=json.loads((RESULTS/'frozen-selection.json').read_text())
    performance_figures(config,frozen);occlusion_analysis(config,frozen)
    print('Scientific figures and descriptive analysis complete.',flush=True)
