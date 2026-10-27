"""Package source, measured evidence, and the final report without the raw dataset."""
from pathlib import Path
import json
import shutil
from zipfile import ZipFile,ZIP_DEFLATED
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecg_project.data import ROOT,sha256,save_json

results=ROOT/'results'
verification=json.loads((results/'evidence-verification.json').read_text())
report_audit=json.loads((results/'report-audit.json').read_text())
assert len(verification['verified_neural_runs'])==12
assert report_audit.get('rendered_page_count')==6 and report_audit.get('all_pages_visually_inspected')
for source,target in [
    (ROOT/'data/prepared/data-audit.json',results/'data-audit.json'),
    (ROOT/'data/prepared/normalization.npz',results/'normalization.npz'),
    (ROOT/'data/raw/ptb-xl-1.0.3/download-provenance.json',results/'data-provenance.json')]:
    if source.is_file():
        shutil.copyfile(source,target)
    elif not target.is_file():
        raise FileNotFoundError('Missing packaged evidence: '+str(target))
files=[ROOT/'README.md',ROOT/'.gitignore',ROOT/'requirements-lock.txt']
files.append(ROOT/'Healthcare_ECG_Colab.ipynb')
files.append(ROOT/'presentation/Healthcare_ECG_CA3_Presentation.pptx')
files.extend([ROOT/'requirements-notebook.txt', ROOT/'.gitattributes', ROOT/'DATASET_LICENSE.txt',
              ROOT/'report/Healthcare_ECG_IEEE_Report.docx'])
for folder in ['ecg_project','configs','tests','scripts']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='probe_data.py')
files.extend(p for p in results.rglob('*') if p.is_file() and p.name not in ['last.pt','bundle-manifest.json'])
files=sorted(set(files))
manifest={'included_files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files},
          'raw_waveforms_included':False,'environment_included':False,'resume_optimizer_states_included':False,
          'best_checkpoints_included':12,'note':'Use the documented downloader to reproduce the checksum-verified dataset.'}
save_json(results/'bundle-manifest.json',manifest)
files.append(results/'bundle-manifest.json')
destination=ROOT/'Healthcare_ECG_Experiments.zip'
with ZipFile(destination,'w',ZIP_DEFLATED,compresslevel=6) as archive:
    for path in files:archive.write(path,'healthcare_ecg/'+path.relative_to(ROOT).as_posix())
with ZipFile(destination) as archive:
    assert archive.testzip() is None
    assert sum(name.endswith('/best.pt') for name in archive.namelist())==12
    assert not any('/.venv/' in name or '/.work/' in name or '/records100/' in name for name in archive.namelist())
print(json.dumps({'archive':str(destination),'bytes':destination.stat().st_size,'source_and_evidence_files':len(files),
                  'sha256':sha256(destination)},indent=2))
