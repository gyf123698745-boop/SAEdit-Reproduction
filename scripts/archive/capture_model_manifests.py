"""Capture pinned official provenance, without storing credentials or signed URLs."""
import json
from pathlib import Path
from huggingface_hub import HfApi

out=Path('experiments');out.mkdir(exist_ok=True)
api=HfApi()
specs=[('black-forest-labs/FLUX.1-dev','3de623fc3c33e44ffbe2bad470d0f45bccf2eb21','flux_official_manifest.json'),
       ('Ronenk94/T5_matryoshka_sae','51c3ac09815815dbbcf7491148507a930e0d7944','sae_official_manifest.json')]
for repo,revision,filename in specs:
    info=api.model_info(repo,revision=revision,files_metadata=True,token=True)
    rows=[]
    for entry in info.siblings:
        name=entry.rfilename
        if 'FLUX' in repo:
            relevant=name=='model_index.json' or (name.startswith(('scheduler/','tokenizer/','tokenizer_2/','text_encoder/','text_encoder_2/','transformer/','vae/')) and name.endswith(('.json','.txt','.model','.safetensors')) and not name.endswith('.fp8.safetensors'))
        else:relevant=name in ['.gitattributes','README.md','matryoshka_sae_top_300/config.json','matryoshka_sae_top_300/pytorch_model.bin']
        if relevant:
            rows.append(dict(file=name,size=entry.size,sha256=entry.lfs.sha256 if entry.lfs else None,git_blob=entry.blob_id))
    (out/filename).write_text(json.dumps(dict(repository=repo,revision=info.sha,files=rows),indent=2))
    print(filename,len(rows),'files')
