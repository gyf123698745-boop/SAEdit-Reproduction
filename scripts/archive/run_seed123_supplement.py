"""Official callback strength sweep with explicit diagnostic variants."""
import json, os, time, hashlib, argparse, shutil
from PIL import Image
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--case",type=int,default=0,choices=[0])
p.add_argument("--seed",type=int,default=123,choices=[123])
args=p.parse_args()
matrix=json.loads(Path("matrix.json").read_text())
case=matrix["cases"][args.case]
from pathlib import Path
import numpy as np
import torch, yaml
from diffusers import FluxPipeline
from src.saedit_cpu import SAEditCallback
from src.models.matryoshka_sae import GlobalBatchTopKMatryoshkaSAE
from assert_model_verified import verify
verified_model=verify()

out=Path('experiments')/('supplement_'+os.environ['SLURM_JOB_ID'])
out.mkdir(parents=True,exist_ok=False)
prompt=case["prompt"]
old=Path("experiments/matrix_3124995")
previous=json.loads((old/"metrics.json").read_text())
assert prompt==previous["prompt"] and args.seed==previous["seed"]
for f in ["model_verification.json","sae_verification.json"]: shutil.copyfile(Path("experiments")/f,out/f)
shutil.copyfile("matrix.json",out/"matrix_config.json")
pipe=FluxPipeline.from_pretrained('models',torch_dtype=torch.bfloat16,local_files_only=True)
pipe.enable_sequential_cpu_offload()
sae=GlobalBatchTopKMatryoshkaSAE.from_pretrained('models/sae/matryoshka_sae_top_300',device='cuda')
pairs=yaml.safe_load((Path("src/variations")/case["variation"]).read_text())['sentence_pairs']
inner=SAEditCallback(pipe,sae,[case["token"]],0.8,pairs,prompt=prompt,max_sequence_length=256)
direction=inner.global_dir_sae.detach().clone()
torch.save(direction.cpu(),out/'direction.pt')
report={'prompt':prompt,'seed':args.seed,'case':case,'resolution':1024,'steps':40,'cfg':3.5,'pair_count':len(pairs),
        'modes':{'official':'all-token reconstruction','target_only':'target-token replacement diagnostic'},'runs':[]}
report['model_revision']=verified_model['revision']
report['model_integrity_verified']=True
report['sae_revision']=verified_model['sae_revision']
report['source_sha256']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in [__file__,'src/saedit_cpu.py','src/models/matryoshka_sae.py','src/models/sae_base.py']}
baseline=np.asarray(Image.open(old/"baseline_0.0.png")).astype(np.float32)
class Wrapper:
    def __init__(self,mode): self.mode=mode
    def __call__(self,p,s,t,k):
        r=inner(p,s,t,k)
        if s in [0,19,39]:
            ids,_=inner.get_idx_to_edit(prompt)
            latent=inner.cache[prompt][:,ids,:]
            scale=float(inner.interpolation_factors[s])
            increment=scale*direction.to(latent.device,dtype=latent.dtype)
            edited=latent+increment
            actual=(edited.float()-latent.float())
            report.setdefault('injection_numerics',[]).append({
                'mode':self.mode,'factor':factor,'step':s,'scale':scale,
                'target_indices':ids,'direction_norm':float(direction.float().norm()),
                'direction_max_abs':float(direction.float().abs().max()),
                'nonzero_direction_features':int(direction.ne(0).sum()),
                'changed_latent_elements':int(actual.ne(0).sum()),
                'actual_increment_norm':float(actual.norm()),
                'requested_increment_norm':float(increment.float().norm())})
        if self.mode=='target_only':
            original=inner.original_prompt_embeds.clone()
            ids,_=inner.get_idx_to_edit(prompt)
            original[:,ids,:]=r['prompt_embeds'][:,ids,:]
            r['prompt_embeds']=original
        return r

tasks=[("official",0),("official",0.2),("official",0.4),("target_only",0)]
for mode,factor in tasks:
    start=time.time(); kw={}
    if mode!='baseline':
        inner.cache={}; inner.original_prompt_embeds=None
        inner.global_dir_sae=direction
        inner.interpolation_factors=inner.calculate_factor(factor)
        kw={'callback_on_step_end':Wrapper(mode),'callback_on_step_end_tensor_inputs':['prompt_embeds']}
    image=pipe(prompt=prompt,height=1024,width=1024,num_inference_steps=40,guidance_scale=3.5,
               max_sequence_length=256,generator=torch.Generator(device='cuda').manual_seed(args.seed),**kw).images[0]
    name=f'{mode}_{factor:.1f}.png'; image.save(out/name)
    (out/(name+".config.json")).write_text(json.dumps(dict(prompt=prompt,seed=args.seed,height=1024,width=1024,steps=40,guidance=3.5,sequence_length=256,mode=mode,factor=factor,pair_count=len(pairs)),indent=2))
    a=np.asarray(image).astype(np.float32)
    if baseline is None: baseline=a.copy()
    report['runs'].append({'file':name,'mode':mode,'factor':factor,'seconds':time.time()-start,
                          'mae_vs_baseline':float(np.abs(a-baseline).mean()),
                          'mse_vs_baseline':float(((a-baseline)**2).mean())})
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print('SAVED',name,flush=True)
(out/'COMPLETED').write_text('success\n')
