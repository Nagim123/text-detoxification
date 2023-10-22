# Text detoxification assignment
## Student
**Name**: Nagim Isyanbaev
<br/>
**Email**: n.isyanbaev@innopolis.university
<br/>
**Group numer**: B21-DS-02
## Installation
Everything was tested using **Python v3.11.3**
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
You can train any model that I considered in this assignment.
<br/>
For example:
```console
python text-detoxification/src/models/train_model.py lstm --epochs 10 --batch_size 64
```
The best and last weights will be saved in *text-detoxification\models* path.
## Prediction
If you trained a model and want to make predictions based on some data, you need to create a **test.txt** file and put the text there (for several texts separate them by a new line). Then you can call prediction script like that:
```console
python text-detoxification/src/models/predict_model.py [lstm|ae_lstm|transformer] your_weight_name.pt path/to/test.txt
```
For metric calculation you need to create additional file **compare.txt** with true detoxification and call script with these flags:
```console
python text-detoxification/src/models/predict_model.py [lstm|ae_lstm|transformer|T5] your_weight_name.pt path/to/test.txt --compare path/to/compare.txt --out_dir output_dir/result.json 
```
After completion you will get json file with all metric calculated for all texts generated by model.
<br/><br/>
Example:
1. Without comparison
```console
python text-detoxification/src/models/predict_model.py lstm lstm.pt test.txt
```
2. With comparison
```console
python text-detoxification/src/models/predict_model.py transformer transformer.pt ../test.txt --compare ../compare.txt --out_dir ../result.json 
```
3. Using T5 (no weights)
```console
python text-detoxification/src/models/predict_model.py T5 _ test.txt
```
*When using T5 as a prediction model, weights will be automatically downloaded from HuggingFace.*

## Download trained models weights
If you do not want to train models you can use models trained by me. There are 2 ways how to get them.
1. Download from google drive ([lstm](https://drive.google.com/file/d/1ZD3Fi51Cmf_lrvlTlUlx0ZTKoWooLkXD/view?usp=share_link), [ae_lstm](https://drive.google.com/file/d/1I4oFZWK7qecLEiTlPuD_euMn89WY_emp/view?usp=share_link), [transformer](https://drive.google.com/file/d/1ilGwJUWX5KKk6caIzPI9wSAoBQ186slv/view?usp=share_link)) and put them into *text-detoxification/models* directory.
2. Use script to automatic download.
```console
python text-detoxification/src/data/download_weights.py
```

## Visualization
For visualization you can use the following command.
```console
python text-detoxification/src/visualization/visualize.py --metric_file path/to/result.json --plot_losses --model_name [lstm|ae_lstm|transformer|T5]
```
* --metric_file : Visualize metrics generated by prediction script
* --plot_losses : Plot average training and validation losses obtained during training by me. (Do not work for T5 model!)
* --model_name: Name of model (lstm, ae_lstm, transformer, T5)
<br/>
Saves figures in *text-detoxification/reports/figures* as .png files.