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
### ParaNMT preprocessing
In case you want to use the ParaNMT dataset, you can use a script to automatically prepare it for training.
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
TODO
## Prediction
TODO