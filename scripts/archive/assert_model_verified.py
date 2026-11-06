"""Refuse experiments without a passing, unchanged model integrity report."""
import json
from pathlib import Path

def verify():
    report=json.loads(Path('experiments/model_verification.json').read_text())
    if not report.get('all_checked_match'):
        raise RuntimeError('Model integrity has not passed; generation prohibited')
    for row in report['files']:
        stat=(Path('models')/row['file']).stat()
        if stat.st_size!=row['actual_size'] or stat.st_mtime_ns!=row.get('actual_mtime_ns'):
            raise RuntimeError('Model changed after verification: '+row['file'])
    sae=json.loads(Path('experiments/sae_verification.json').read_text())
    if not sae.get('all_checked_match'): raise RuntimeError('SAE integrity has not passed')
    for row in sae['files']:
        stat=(Path('models/sae')/row['file']).stat()
        if stat.st_size!=row['size'] or stat.st_mtime_ns!=row.get('actual_mtime_ns'):
            raise RuntimeError('SAE changed after verification: '+row['file'])
    report['sae_revision']=sae['revision']
    return report

if __name__=='__main__':
    print('MODEL VERIFIED',verify()['revision'])
