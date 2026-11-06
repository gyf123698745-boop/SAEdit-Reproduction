"""One plain FLUX image and non-generative replay of the existing callback."""
import os,json,time,hashlib,shutil
from pathlib import Path
import torch
from diffusers import FluxPipeline
from src.saedit_cpu import SAEditCallback
from src.models.matryoshka_sae import GlobalBatchTopKMatryoshkaSAE
from assert_model_verified import verify

out=Path('experiments')/('bald_diagnosis_'+os.environ['SLURM_JOB_ID'])
out.mkdir(exist_ok=False)
verification=verify()
for name in ['model_verification.json','sae_verification.json']:
    shutil.copyfile(Path('experiments')/name,out/name)
hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['src/saedit_cpu.py','src/saedit.py','src/models/matryoshka_sae.py','src/variations/bald_man.yaml',__file__]}
assert hashes['src/saedit_cpu.py']=='005247c164dec3d696d9f98ae7a74b02beb5d8c12b7c499d3c63f07f8c5b0682'
pipe=FluxPipeline.from_pretrained('models',torch_dtype=torch.bfloat16,local_files_only=True)
pipe.enable_sequential_cpu_offload()
prompt='a portrait photo of a man in a garden'
target='a portrait photo of a bald man in a garden'
config=dict(prompt=target,seed=123,height=1024,width=1024,num_inference_steps=40,guidance_scale=3.5,max_sequence_length=256,callback=None,dtype='bfloat16',offload='sequential')
(out/'direct_bald.config.json').write_text(json.dumps(config,indent=2))
start=time.time()
with torch.inference_mode():
    image=pipe(prompt=target,height=1024,width=1024,num_inference_steps=40,guidance_scale=3.5,max_sequence_length=256,generator=torch.Generator(device='cuda').manual_seed(123)).images[0]
image.save(out/'direct_bald.png')
print('DIRECT IMAGE SAVED',time.time()-start,flush=True)
with torch.inference_mode():
    embeds=pipe._get_t5_prompt_embeds(prompt=[prompt],max_sequence_length=256)
sae=GlobalBatchTopKMatryoshkaSAE.from_pretrained('models/sae/matryoshka_sae_top_300',device='cuda').to(torch.bfloat16)
def stats(x):
    a=x.detach().float()
    return dict(shape=list(a.shape),dtype=str(x.dtype),norm=float(a.norm()),nonzero=int(a.ne(0).sum()),nan=int(a.isnan().sum()),inf=int(a.isinf().sum()),max_abs=float(a.abs().max()),negative=int(a.lt(0).sum()))
bald=torch.load('experiments/bald_3126203/direction.pt',map_location='cuda',weights_only=True)
smile=torch.load('experiments/matrix_3124995/direction.pt',map_location='cuda',weights_only=True)
report=dict(model_revision=verification['revision'],source_sha256=hashes,plain_generation_seconds=time.time()-start,directions={'bald':stats(bald),'smile':stats(smile)},direction_comparison={'cosine':float(torch.nn.functional.cosine_similarity(bald.float(),smile.float(),dim=0)),'difference_norm':float((bald.float()-smile.float()).norm())},replay=[])
# Use the unchanged production callback with saved direction; skip only its costly direction constructor.
cb=SAEditCallback.__new__(SAEditCallback)
cb.sae=sae;cb.pipeline=pipe;cb.prompt=prompt;cb.source_tokens_to_edit=['man'];cb.global_dir_sae=bald;cb.max_sequence_length=256
ids,tokens=cb.get_idx_to_edit(prompt)
assert ids==[7]
report['tokens']=tokens;report['target_indices']=ids;report['sae_training']=sae.training
with torch.inference_mode():
    latent=sae.encode(embeds.cuda());zero=sae.decode(latent).to(embeds.device,dtype=embeds.dtype)
    report['original_target_latent']=stats(latent[:,ids,:])
    report['reconstruction_target_delta']=stats((zero[:,ids,:].float()-embeds[:,ids,:].float()))
    non=[i for i in range(256) if i not in ids]
    for factor in [0.4,0.8]:
        cb.cache={};cb.original_prompt_embeds=None;cb.interpolation_factors=cb.calculate_factor(factor)
        # First callback is essential to reproduce initialization and cache state.
        for step in range(40):
            result=cb(pipe,step,None,{'prompt_embeds':embeds})['prompt_embeds']
            base=cb.cache[prompt][:,ids,:]
            requested=float(cb.interpolation_factors[step])*bald.to(base.dtype)
            delta=(base+requested).float()-base.float()
            decoded=result.float()-zero.float()
            report['replay'].append(dict(factor=factor,callback_index=step,scale=float(cb.interpolation_factors[step]),used_by_next_denoise=step<39,actual_latent_delta=stats(delta),target_embedding_delta=stats(decoded[:,ids,:]),nontarget_embedding_delta=stats(decoded[:,non,:])))
(out/'diagnosis.json').write_text(json.dumps(report,indent=2))
(out/'COMPLETED').write_text('success\n')
print('DIAGNOSIS COMPLETED',flush=True)
