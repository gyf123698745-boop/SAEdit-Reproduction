"""Verify local FLUX weights against captured official HF LFS hashes."""
import hashlib,json
from pathlib import Path
manifest=json.loads(Path('experiments/flux_official_manifest.json').read_text())
results=[]
for row in manifest['files']:
    path=Path('models')/row['file']
    size=path.stat().st_size
    expected=row['sha256']
    actual=None
    if expected:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
        actual=digest.hexdigest()
    blob=None
    if not expected and row.get('git_blob'):
        blob=hashlib.sha1(b'blob '+str(size).encode()+b'\0'+path.read_bytes()).hexdigest()
    ok=size==row['size'] and (expected is None or actual==expected) and (blob is None or blob==row['git_blob'])
    results.append(dict(**row,actual_size=size,actual_sha256=actual,actual_mtime_ns=path.stat().st_mtime_ns,ok=ok))
    print(row['file'],ok,flush=True)
report=dict(revision=manifest['revision'],files=results,
            all_checked_match=all(r['ok'] for r in results))
Path('experiments/model_verification.json').write_text(json.dumps(report,indent=2))
if not report['all_checked_match']: raise RuntimeError('Model files mismatch')
sae_manifest_path=Path('experiments/sae_official_manifest.json')
if not sae_manifest_path.exists(): raise FileNotFoundError('SAE official manifest required')
sae_manifest=json.loads(sae_manifest_path.read_text())
sae_results=[]
for row in sae_manifest['files']:
    path=Path('models/sae')/row['file']
    digest=hashlib.sha256()
    if row['sha256']:
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    actual=digest.hexdigest() if row['sha256'] else None
    blob=None
    if not row['sha256'] and row.get('git_blob'):
        blob=hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0'+path.read_bytes()).hexdigest()
    ok=path.stat().st_size==row['size'] and (row['sha256'] is None or actual==row['sha256']) and (blob is None or blob==row['git_blob'])
    sae_results.append(dict(**row,actual_sha256=actual,actual_mtime_ns=path.stat().st_mtime_ns,ok=ok))
sae_report=dict(revision=sae_manifest['revision'],files=sae_results,all_checked_match=all(r['ok'] for r in sae_results))
Path('experiments/sae_verification.json').write_text(json.dumps(sae_report,indent=2))
if not sae_report['all_checked_match']: raise RuntimeError('SAE model mismatch')
