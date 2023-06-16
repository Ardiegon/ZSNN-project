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
MAIN_DATASET_DIR_PATH = "path/to/your/dataset"
```
***Last file of the path should be 'Console_sliced'***

Check if everything is working with
```bash
python src/scripts/train.py --model DummyModel
```

## Use 
If you want to use pretreined model weights in generation (recommended) and training add in specific config file weights path:
```
"checkpoint_path": "src/configs/weights/image_generator.pth"
```

### Training

HuggingFace generator train:
```bash
 python src/scripts/train.py -m UNet2DModelAdapted -c src/configs/models/default_UNet2DModelAdapted.json
```

ConditionalModel train:
```bash
python src/scripts/train.py -m ConditionModel -c src/configs/models/default_conditional_model.json 
```

HuggingFace generator train:
```bash
python src/scripts/train.py -m AggregatedModel -c src/configs/models/default_aggregated_model.json
```

### Generation
```bash
python src/scripts/generate.py -m UNet2DModelAdapted -c src/configs/models/image_generator.json
```


###  Do wytestowania

### Wytestowane
- Pretrening na datasetach tekstur przemysłowych, na CIFAR 10 
- dodanie regularyzacji L2
- przestrajanie parametrów
- Zamiana funkcji kosztu na MSE
- Pretrening na Motylach
- Model wyostrzający na wyjściu preuczonego brancha

