"""CPU-only analysis of eight existing validated images. No generation."""
import json,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch,lpips
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
metric=lpips.LPIPS(net='alex',version='0.1').eval()
def load(p):
    a=np.array(Image.open(p).convert('RGB'),copy=True)
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/127.5-1
report={'metric':'LPIPS AlexNet v0.1','resolution':'native1024','normalization':'RGB [-1,1]','device':'CPU','results':[]}
with torch.inference_mode():
    for job in ['3124956','3124995']:
        directory=Path('experiments')/('matrix_'+job)
        meta=json.loads((directory/'metrics.json').read_text())
        assert meta['model_integrity_verified'] and len(meta['runs'])==4
        images={r['file']:load(directory/r['file']) for r in meta['runs']}
        pairs=[('baseline_0.0.png',r['file']) for r in meta['runs']]
        pairs.append(('official_0.0.png','official_0.8.png'))
        for reference,image in pairs:
            assert images[reference].shape==images[image].shape
            value=float(metric(images[reference],images[image]).item())
            report['results'].append(dict(job=job,seed=meta['seed'],reference=reference,image=image,lpips=value))
Path('experiments/existing_eight_analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
