# NGC-PC-Transformers
Implementation of a Predictive Coding Transformer built on the Neural Generative Coding framework.

## Installation

Clone the Repository

```bash
git clone https://github.com/iCog-Labs-Dev/NGC-PC-Transformers.git
```
Navigate to the Project Directory

```bash
cd NGC-PC-Transformers
```
Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
``` 
Install required packages:

```bash
pip install -r requirements.txt
```
Run the tokenization
```bash
python -m data_preprocess.tokenizer
```
Run the training
```bash
python train.py
```
Run the evaluation
```bash
python eval.py
```
Run the generation
```bash
python generation.py
``` 
