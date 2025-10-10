# LLM-refusal-analysis
Class project for Georgia Tech's CS 4650: investigating monolith vs modular refusal behavior of LLMS across refusal categories 

## Environment setup

### Setting up a Conda Environment in Scratch (PACE-ICE)

#### 1. Load the Anaconda Module
PACE provides Anaconda as a preinstalled module.

```bash
module load anaconda3
```

> Check available versions with:
> ```bash
> module avail anaconda
> ```

#### 2. Create a Conda Environment in Scratch

By default, `conda` environments go to your home directory (`~/.conda/envs`),  
which has limited quota. To avoid filling it up, create a custom environment directory in scratch:

```bash
conda env create --prefix /scratch/<path_to_env_parent_dir>/llm_refusal_env --file llm_refusal_env.yml
```

Alternatively, you could create the environment from scratch without using the yml file:

```bash
# Create the environment explicitly in scratch
conda create --prefix /scratch/<path_to_env_parent_dir>/llm_refusal_env python=3.11 -y
```

#### 3. Activate the Environment

```bash
conda activate /scratch/<path_to_env_parent_dir>/llm_refusal_env
```

> You **must use the full path** when activating environments created outside your home directory.

#### 4. Install Required Packages

Once activated, install any packages you need, such as PyTorch, OpenCV, etc.

```bash
# Example for PyTorch + utilities
conda install pytorch torchvision torchaudio -c pytorch -y
conda install numpy matplotlib opencv scipy scikit-learn tqdm -y
conda install jupyterlab -y
```

If you created the environment from scratch, use the requirements.txt file:

```bash
pip install requirements.txt
```

#### 5. Verify Installation

```bash
python -m pip list
python -c "import torch; print(torch.__version__)"
```

#### 6. (Optional) Export and Reuse the Environment

You can save your environment spec for reproducibility:

```bash
conda env export > llm_refusal_env.yml
```

and recreate it later with:

```bash
conda env create --prefix /scratch/<path_to_env_parent_dir>/llm_refusal_env --file llm_refusal_env.yml
```

#### 8. Deactivate Environment

```bash
conda deactivate
```

### Model Download Instructions

#### 1. Login to Hugging Face
```bash
huggingface-cli login
# This will prompt you for your Hugging Face API token
``` 

#### 2. Set a scratch folder for model downloads
```bash
export HF_HOME=/home/hice1/vkulkarni46/scratch/huggingface
```

#### 3. Download LLaMA-2-7B-Chat
```bash
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --cache-dir $HF_HOME
```
#### 5. Download Mistral-7B-Instruct
```bash
huggingface-cli download mistal-instruct/mistral-7b --cache-dir $HF_HOME
```
