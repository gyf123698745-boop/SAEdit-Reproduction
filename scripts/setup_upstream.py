"""Install only the portable runner and integrity tools; preserve upstream source."""
import argparse,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('upstream',type=Path);a=p.parse_args()
root=a.upstream.resolve();here=Path(__file__).resolve().parent
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if commit != 'fc6e6704195375b06fb97cd7cc7389087cad19b9':
    p.error('Check out the pinned upstream commit before running setup')
def put(path,text):
    if path.exists() and path.read_text(encoding='utf-8')!=text:raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8',newline='\n')
source=(root/'src/saedit.py').read_text(encoding='utf-8')
old='self.sae.decode(a_src).to(prompt_embeds.dtype)'
if source.count(old) != 1:
    p.error('The upstream callback does not match the expected device adaptation')
put(root/'src/saedit_cpu.py',source.replace(old,'self.sae.decode(a_src).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)'))
for name in ['run_case.py','assert_model_verified.py','verify_model_files.py','capture_model_manifests.py']:
    path=here/name if name=='run_case.py' else here/'archive'/name
    put(root/'examples'/name,path.read_text(encoding='utf-8'))
print('Prepared runner; upstream core source unchanged. No experiments submitted.')
