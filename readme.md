# ZSNN Projekt
Image generation with diffusion models

## Instalation
For Windows user paste following code

```bash
conda env create -f env.yml
conda activate zsnn-env
conda develop src 
```

For Linux user paste following code:
```bash
conda create --name=zsnn-env
conda activate zsnn-env
pip install -r requirements.txt
conda install conda-build
conda develop src .
```

## Setup
Fulfill dateset path inside *src/configs/path.py*:
```python
DATASET_DIR_PATH = "path/to/your/dataset"
```
***Last file of the path should be 'Console_sliced'***

Check if everything is working with
```bash
python src/scripts/train.py --model DummyModel
```