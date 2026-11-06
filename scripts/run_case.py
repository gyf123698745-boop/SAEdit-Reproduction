"""Portable entry point; same official callback, optional target-only wrapper.

Packaging-only addition: syntax checked, not used to generate the archived results.
Run from the pinned upstream checkout after setup_upstream.py.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--attribute',choices=['smiling','baldness'],required=True)
p.add_argument('--mode',choices=['baseline','official','target_only','direct_prompt'],required=True)
p.add_argument('--factor',type=float,default=0)
p.add_argument('--seed',type=int,default=123)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if not math.isfinite(a.factor) or a.factor < 0:
    p.error('Factor must be finite and nonnegative')
if a.mode in ['baseline','direct_prompt'] and a.factor!=0:p.error('No injection factor for a no-callback run')
if a.output.exists():raise FileExistsError('Refusing to overwrite '+str(a.output))
import torch,yaml
from diffusers import FluxPipeline
from assert_model_verified import verify
verification=verify()
prompt='a portrait photo of a man in a garden'
if a.mode=='direct_prompt':
    prompt='a portrait photo of a '+('bald' if a.attribute=='baldness' else 'smiling')+' man in a garden'
pipe=FluxPipeline.from_pretrained('models',torch_dtype=torch.bfloat16,local_files_only=True)
pipe.enable_sequential_cpu_offload()
kw={};config=dict(prompt=prompt,seed=a.seed,factor=a.factor,mode=a.mode,attribute=a.attribute,width=1024,height=1024,steps=40,guidance=3.5,sequence_length=256,dtype='bfloat16',offload='sequential_cpu',model_revision=verification['revision'],sae_revision=verification['sae_revision'])
a.output.mkdir(parents=True)
if a.mode in ['official','target_only']:
    from src.saedit_cpu import SAEditCallback
    from src.models.matryoshka_sae import GlobalBatchTopKMatryoshkaSAE
    variation=Path('src/variations')/('bald_man.yaml' if a.attribute=='baldness' else 'smiling_man.yaml')
    pairs=yaml.safe_load(variation.read_text())['sentence_pairs']
    sae=GlobalBatchTopKMatryoshkaSAE.from_pretrained('models/sae/matryoshka_sae_top_300',device='cuda')
    inner=SAEditCallback(pipe,sae,['man'],a.factor,pairs,prompt=prompt,max_sequence_length=256)
    ids,_=inner.get_idx_to_edit(prompt)
    assert ids==[7],f'Unexpected token indices: {ids}'
    config.update(target_indices=ids,variation=str(variation),pair_count=len(pairs),yaml_sha256=hashlib.sha256(variation.read_bytes()).hexdigest())
    torch.save(inner.global_dir_sae.cpu(),a.output/'direction.pt')
    def callback(pipe,step,timestep,kwargs):
        result=inner(pipe,step,timestep,kwargs)
        if a.mode=='target_only':
            original=inner.original_prompt_embeds.clone()
            original[:,ids,:]=result['prompt_embeds'][:,ids,:]
            result['prompt_embeds']=original
        return result
    kw=dict(callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['prompt_embeds'])
(a.output/'config.json').write_text(json.dumps(config,indent=2))
start=time.time()
image=pipe(prompt=prompt,height=1024,width=1024,num_inference_steps=40,guidance_scale=3.5,max_sequence_length=256,generator=torch.Generator(device='cuda').manual_seed(a.seed),**kw).images[0]
image.save(a.output/'image.png')
(a.output/'timing.json').write_text(json.dumps({'generation_seconds':time.time()-start}))
