# Text detoxification assignment
## Student
**Name**: Nagim Isyanbaev
<br/>
**Email**: n.isyanbaev@innopolis.university
<br/>
**Group numer**: B21-DS-02
## Installation
#### 1. Download repository
```console
git clone https://github.com/Nagim123/text-detoxification.git
```
#### 2. Create virtual environment and activate it (OPTIONAL)
```console
python -m venv venv
```
Windows:
```console
venv\Scripts\activate
```
Linux:
```console
source ./venv/bin/activate
```
#### 3. Install python libraries
```console
pip install -r requirements.txt
```
#### 4. Download English word tokenizator
```console
python -m spacy download en_core_web_sm
```
## Data transformation
For training or prediction you need a vocabulary file. Moreover for training you need preprocessed dataset. There are two ways to generate these files.
### ParaNMT preprocessing
In case you want to use the ParaNMT dataset, you can use a script to automatically prepare it for training and generate vocabulary.
```console
python text-detoxification/src/data/make_dataset.py --logging
```
You can remove the *--logging* flag if you don't want to see the preprocessing progress.
### Custom data preprocessing
In case you want to use your own data, follow the instruction below:
1. Create two files **toxic.txt** and **detoxified.txt**.
2. In **toxic.txt** place toxic texts separated by new line.
3. In **detoxified.txt** place detoxified text versions of the same texts from **toxic.txt** separated by new line.
4. Call script to prepare data and create vocabulary.
```console
python text-detoxification/src/data/preprocess_texts.py toxic.txt dataset.pt --translated_text_file detoxified.txt --vocab_encode vocab.pt [OPTIONAL: --logging]
```
## Training
To train a model you need **dataset.pt** and **vocab.pt** files. Read *Data transformation* section to know how to generate these files. <br/>
To train a model you need to call training script.
```console
python text-detoxification/src/models/train_model.py [lstm|ae_lstm|transformer] --epochs EPOCH_NUM --batch_size BATCH_SIZE
```
For example:
```console
python text-detoxification/src/models/train_model.py lstm --epochs 10 --batch_size 64
```
The best and last weights will be saved in *text-detoxification\models* path.
## Prediction
TODO
## Visualization