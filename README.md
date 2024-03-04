# Investigating Capabilities of Vision-Language Models in Understanding Relationships between Subjects, Objects and Transitive Verbs
The goal of this project was that of measuring the capabilities of various multimodal visual-language models of understanding the relationship between subjects and objects that depend from a transitive verb.
You can find a detailed description of our work [here](https://drive.google.com/file/d/1Iw0JbckMY9aJNIdrapcn7CNfK47r0uys/view?usp=sharing).

---
# Setup
## 1. Create and Activate the Conda Environment
```
conda create -n t_verbs_vl_eval -y python=3.8
conda activate t_verbs_vl_eval
```
## 2. Install the dependencies
```
pip install -r requirements.txt
```
## 3. Set the path
```
export PYTHONPATH=/path/to/your/project/folder
cd experiments
```
## 4. Run the experiments
### 4.1 Run the plausibility bias experiment
Run the preliminary experiment performed to measure the impact of nonsensical foils on the two datasets. You don't have to worry about getting the models: they will download automatically if not presente in the project folder already.
The brackets indicate the possible values that can be assigned to the argument.

```
python -m zero_shot --model=['ALBEF','XVLM','BLIP','X2VLM', 'NegCLIP'] --experiment=pre --dataset=['VALSE', 'ARO','all']
```
### 4.1 Run the ITM experiments (with both aggregated results as well as computed by category)
```
python -m zero_shot --model=['ALBEF','XVLM','BLIP','X2VLM', 'NegCLIP'] --experiment=itm --dataset=['VALSE', 'ARO','all']
```
