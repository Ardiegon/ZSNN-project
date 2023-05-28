# ZSNN Projekt
Image generation with diffusion models

## Instalation
```bash
conda env create -f env.yml
conda activate zsnn-env
conda develop src 
```
Check if everything is working with
```bash
python src/scripts/train.py --model DummyModel
```