"""Throughput calibration on generated inputs; never treated as model evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time
import torch
from ecg_project.models import build_model,parameter_count
from ecg_project.data import save_json,ROOT

results=[]
for threads in [1,4,8]:
    torch.set_num_threads(threads)
    for name in ['compact_cnn','resnet1d','inception_time']:
        torch.manual_seed(42)
        model=build_model(name)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
        x=torch.randn(64,12,1000);y=torch.randint(0,2,(64,5)).float()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);torch.nn.functional.binary_cross_entropy_with_logits(model(x),y).backward();optimizer.step()
        started=time.perf_counter()
        for step in range(8):
            optimizer.zero_grad(set_to_none=True);torch.nn.functional.binary_cross_entropy_with_logits(model(x),y).backward();optimizer.step()
        seconds=(time.perf_counter()-started)/8
        row={'model':name,'threads':threads,'batch_size':64,'seconds_per_training_step':seconds,
             'estimated_seconds_per_epoch_17400_records':seconds*272,'parameters':parameter_count(model),
             'purpose':'synthetic-input timing only; no performance metrics'}
        results.append(row);print(json.dumps(row),flush=True)
save_json(ROOT/'results'/'hardware-throughput.json',results)
