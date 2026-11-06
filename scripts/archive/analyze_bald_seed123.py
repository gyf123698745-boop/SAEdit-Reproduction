import sys,json,hashlib
from pathlib import Path
import torch,lpips,numpy as np
from PIL import Image
torch.set_num_threads(1)
p=Path('experiments')/('bald_'+sys.argv[1]); old=Path('experiments/matrix_3124995')
paths={'baseline':old/'baseline_0.0.png','old_official0':old/'official_0.0.png',**{'official'+str(f):p/f'official_{f:.1f}.png' for f in [0,0.4,0.8]}}
arrays={k:np.array(Image.open(v).convert('RGB'),copy=True) for k,v in paths.items()}
t={k:torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/127.5-1 for k,a in arrays.items()}
m=lpips.LPIPS(net='alex',version='0.1').eval()
pairs=[('old_official0','official0'),('baseline','official0'),('baseline','official0.4'),('baseline','official0.8'),('official0','official0'),('official0','official0.4'),('official0','official0.8'),('official0.4','official0.8')]
rows=[]
with torch.inference_mode():
 for a,b in pairs:
    rows.append(dict(reference=a,image=b,lpips=float(m(t[a],t[b]).item()),pixel_equal=bool(np.array_equal(arrays[a],arrays[b])),mae=float(np.abs(arrays[a].astype(float)-arrays[b].astype(float)).mean())))
report=dict(metric='LPIPS AlexNet v0.1',resolution='native1024',normalization='RGB [-1,1]',pairs=rows,files={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in paths.items()})
(p/'pairwise_analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
