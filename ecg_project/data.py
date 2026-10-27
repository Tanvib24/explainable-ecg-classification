from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import wfdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'ptb-xl-1.0.3'
CACHE = ROOT / 'data' / 'prepared'
BASE = 'https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
        LOCAL.session.mount('https://', HTTPAdapter(max_retries=retry))
    return LOCAL.session


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def download_file(relative, expected=None):
    target = RAW / relative
    if target.is_file() and (not expected or sha256(target)==expected): return 'cached'
    target.parent.mkdir(parents=True,exist_ok=True)
    temp = target.with_suffix(target.suffix+'.partial')
    for attempt in range(4):
        try:
            response = session().get(BASE+relative,timeout=(15,60))
            response.raise_for_status()
            temp.write_bytes(response.content)
            if expected and sha256(temp)!=expected: raise ValueError('SHA256 mismatch: '+relative)
            temp.replace(target)
            return 'downloaded'
        except Exception:
            if attempt==3: raise
            time.sleep(2**attempt)


def fetch(workers=24):
    RAW.mkdir(parents=True,exist_ok=True)
    # The public S3 mirror is explicitly documented by PhysioNet for this release.
    for relative in ['SHA256SUMS.txt','ptbxl_database.csv','scp_statements.csv','LICENSE.txt']:
        download_file(relative)
    expected={}
    for line in (RAW/'SHA256SUMS.txt').read_text().splitlines():
        digest,relative=line.split(maxsplit=1)
        expected[relative.lstrip('*').removeprefix('./')]=digest
    for relative in ['ptbxl_database.csv','scp_statements.csv','LICENSE.txt']:
        if relative not in expected or sha256(RAW/relative)!=expected[relative]:
            raise ValueError('Metadata checksum mismatch: '+relative)
    meta=pd.read_csv(RAW/'ptbxl_database.csv')
    files=[str(name)+extension for name in meta.filename_lr for extension in ['.hea','.dat']]
    if any(name not in expected for name in files): raise ValueError('Manifest is missing requested waveforms')
    started=time.time();last=started
    print(f'Fetching {len(files)} waveform files from the official public S3 mirror',flush=True)
    completed=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(download_file,name,expected[name]):name for name in files}
        for future in as_completed(futures):
            future.result();completed+=1
            if time.time()-last>20 or completed==len(files):
                print(f'Download {completed}/{len(files)}; elapsed {time.time()-started:.0f}s',flush=True);last=time.time()
    save_json(RAW/'download-provenance.json',{
        'version':'1.0.3','source':BASE,'official_page':'https://physionet.org/content/ptb-xl/1.0.3/',
        'files_verified':len(files),'manifest_sha256':sha256(RAW/'SHA256SUMS.txt'),
        'metadata_sha256':sha256(RAW/'ptbxl_database.csv'),'lookup_sha256':sha256(RAW/'scp_statements.csv'),
        'download_seconds':time.time()-started,'waveform_resolution_hz':100})


def partition_masks(meta):
    masks={'train':meta.strat_fold.to_numpy()<=8,'validation':meta.strat_fold.to_numpy()==9,'test':meta.strat_fold.to_numpy()==10}
    assert all(mask.any() for mask in masks.values())
    groups={name:set(meta.loc[mask,'patient_id']) for name,mask in masks.items()}
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        if groups[left]&groups[right]: raise ValueError('Patient leakage between '+left+' and '+right)
    return masks


def prepare():
    CACHE.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(RAW/'ptbxl_database.csv')
    lookup=pd.read_csv(RAW/'scp_statements.csv',index_col=0)
    mapping=lookup.loc[lookup.diagnostic==1,'diagnostic_class'].dropna().to_dict()
    labels=[];keep=[];excluded=[];unknown=set()
    for i,row in meta.iterrows():
        codes=ast.literal_eval(row.scp_codes)
        unknown.update(set(codes)-set(lookup.index))
        categories={mapping[code] for code in codes if code in mapping}
        y=[int(label in categories) for label in CLASSES]
        if any(y):keep.append(i);labels.append(y)
        else:excluded.append({'ecg_id':int(row.ecg_id),'reason':'no_mapped_diagnostic_superclass'})
    if unknown:raise ValueError('Unknown annotation codes: '+str(unknown))
    meta=meta.loc[keep].reset_index(drop=True)
    masks=partition_masks(meta)
    y=np.asarray(labels,dtype=np.float32)
    x=np.lib.format.open_memmap(CACHE/'signals.npy',mode='w+',dtype='float32',shape=(len(meta),12,1000))
    hashes={};duplicate_groups=[];started=time.time()
    for i,row in meta.iterrows():
        waveform,fields=wfdb.rdsamp(str(RAW/row.filename_lr))
        if waveform.shape!=(1000,12) or fields['fs']!=100 or not np.isfinite(waveform).all():
            raise ValueError('Invalid waveform: '+str(row.ecg_id))
        if [lead.upper() for lead in fields['sig_name']]!=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']:
            raise ValueError('Unexpected lead order: '+str(fields['sig_name']))
        values=np.ascontiguousarray(waveform.T,dtype=np.float32)
        digest=hashlib.sha256(values.tobytes()).hexdigest()
        if digest in hashes:
            previous=hashes[digest]
            duplicate_groups.append([int(meta.iloc[previous].ecg_id),int(row.ecg_id)])
            previous_split=next(name for name,mask in masks.items() if mask[previous])
            split=next(name for name,mask in masks.items() if mask[i])
            if previous_split!=split:raise ValueError('Exact waveform leakage between partitions')
        else:hashes[digest]=i
        x[i]=values
        if i%2000==0:print(f'Prepared {i}/{len(meta)}; elapsed {time.time()-started:.0f}s',flush=True)
    x.flush()
    # Double-precision reductions over training data, independent of held-out data.
    total=np.zeros(12,dtype=np.float64);square=np.zeros(12,dtype=np.float64);count=0
    train_indices=np.flatnonzero(masks['train'])
    for start in range(0,len(train_indices),256):
        values=np.asarray(x[train_indices[start:start+256]],dtype=np.float64)
        total+=values.sum(axis=(0,2));square+=(values*values).sum(axis=(0,2));count+=values.shape[0]*1000
    mean=total/count;std=np.sqrt(np.maximum(square/count-mean*mean,0))
    pos=y[masks['train']].sum(axis=0);weights=(len(train_indices)-pos)/pos
    np.save(CACHE/'labels.npy',y)
    meta.to_csv(CACHE/'cohort.csv',index=False)
    np.savez(CACHE/'normalization.npz',mean=mean.astype(np.float32),std=std.astype(np.float32),pos_weight=weights.astype(np.float32))
    summary={'original_records':21799,'eligible_records':len(meta),'excluded':excluded,'classes':CLASSES,
             'patient_disjoint':True,'cross_partition_exact_duplicates':0,'within_partition_exact_duplicates':duplicate_groups,
             'label_cooccurrence':(y.T@y).astype(int).tolist(),'splits':{},'preparation_seconds':time.time()-started,
             'signal_sha256':sha256(CACHE/'signals.npy'),'label_sha256':sha256(CACHE/'labels.npy'),
             'normalization_fit_partition':'train','normalization_epsilon':1e-6,'class_weights':weights.tolist()}
    for name,mask in masks.items():
        summary['splits'][name]={'recordings':int(mask.sum()),'patients':int(meta.loc[mask,'patient_id'].nunique()),
            'positive_labels':dict(zip(CLASSES,y[mask].sum(axis=0).astype(int).tolist()))}
    save_json(CACHE/'data-audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['excluded','within_partition_exact_duplicates']},indent=2),flush=True)


def load_data():
    meta=pd.read_csv(CACHE/'cohort.csv')
    masks=partition_masks(meta)
    x=np.load(CACHE/'signals.npy',mmap_mode='r')
    y=np.load(CACHE/'labels.npy')
    stats=np.load(CACHE/'normalization.npz')
    return x,y,meta,masks,stats


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fetch','prepare']);parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    if args.action=='fetch':fetch(args.workers)
    else:prepare()
