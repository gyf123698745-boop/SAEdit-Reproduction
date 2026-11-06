import sys,json,os,hashlib
from pathlib import Path
import numpy as np
import torch,lpips
from PIL import Image
torch.set_num_threads(1)
new=Path('experiments')/('supplement_'+sys.argv[1]);old=Path('experiments/matrix_3124995')
paths={'baseline':old/'baseline_0.0.png','old_official0':old/'official_0.0.png','official0':new/'official_0.0.png','official0.2':new/'official_0.2.png','official0.4':new/'official_0.4.png','official0.8':old/'official_0.8.png','target0':new/'target_only_0.0.png','target0.8':old/'target_only_0.8.png'}
arrays={k:np.array(Image.open(p).convert('RGB'),copy=True) for k,p in paths.items()}
tensors={k:torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/127.5-1 for k,a in arrays.items()}
metric=lpips.LPIPS(net='alex',version='0.1').eval()
pairs=[('old_official0','official0')]+[('baseline',k) for k in ['official0','official0.2','official0.4','official0.8','target0','target0.8']]+[('official0',k) for k in ['official0.2','official0.4','official0.8']]+[('official0.2','official0.4'),('official0.4','official0.8'),('target0','target0.8')]
rows=[]
with torch.inference_mode():
 for a,b in pairs:
    delta=arrays[a].astype(float)-arrays[b].astype(float)
    rows.append(dict(reference=a,image=b,lpips=float(metric(tensors[a],tensors[b]).item()),pixel_equal=bool(np.array_equal(arrays[a],arrays[b])),mae=float(np.abs(delta).mean()),max_abs=float(np.abs(delta).max())))
report=dict(metric='LPIPS AlexNet v0.1',normalization='RGB [-1,1]',resolution='native1024',pairs=rows,files={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()})
old_direction=torch.load(old/'direction.pt',map_location='cpu',weights_only=True)
new_direction=torch.load(new/'direction.pt',map_location='cpu',weights_only=True)
report['direction_equal']=bool(torch.equal(old_direction,new_direction))
report['direction_max_abs_difference']=float((old_direction.float()-new_direction.float()).abs().max())
(new/'pairwise_analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
