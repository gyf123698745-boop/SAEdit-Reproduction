# Attribution and licensing status

The original SAEdit method and implementation are by Ronen Kamenetsky, Sara Dorfman, Daniel Garibi, Roni Paiss, Or Patashnik and Daniel Cohen-Or.

Official source: https://github.com/ronen94/SAEdit, pinned commit `fc6e6704195375b06fb97cd7cc7389087cad19b9`.

The inspected upstream Git tree contains **no LICENSE file**. The absence of a license file does not grant redistribution rights. Official core source, training code, Prompt Pair YAMLs and weights are not vendored here. Reproduction commands use a separate official checkout; config names and hashes preserve provenance. Original upstream acknowledgements identify diffusers and Matryoshka SAE as dependencies.

Local additions comprise experiment runners, a device-placement adaptation, the Target-only diagnostic wrapper, analysis, presentation and records. Target-only is not an official SAEdit ablation. This public repository does not apply a blanket license to its contents. Dependencies and model weights retain their respective terms; no model weights are redistributed.

Full-resolution generated images and numerical records are independent reproduction artifacts. They are neither official paper figures nor evidence of a complete benchmark reproduction.

## Generated outputs in this public reproduction

All displayed portraits are AI-generated with FLUX.1-dev, including the SAEdit-conditioned outputs. The [model license at the exact recorded revision](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/LICENSE.md), sections 1(a), 2(d) and 2(e), distinguishes outputs from model derivatives and permits output use subject to its stated restrictions and review/disclosure requirements. The included generic portrait outputs have been visually reviewed; their prompts do not name an actual individual. This permission does not grant rights to redistribute model weights, third-party code or any third-party identity. No such weights or upstream core source are included. No blanket software or image license is added here.
