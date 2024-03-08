# ML Project

## Create a Virtual Environment
Install Anaconda.
Using VS Code, type the following commands:

```
conda create -p venv python==3.10 -y
conda activate venv/
conda install ipykernel

# Check Virtual Environments
conda info --envs
```

## Sync project to GitHub
```
git pull
git add .
git commit -m "adding readme notes"
git push -u origin main

# Check status
git status
git remote -v

# Remove files
git rm -r folder_name
## Example:
git rm -r --cached venv

```

## Install requirements
```
 pip install -r requirements.txt
```

## Project Scope
- Problem Definition
- Data Collection
- Data Preparation
- EDA
- Data Pre-Processing
- Model Training
- Choose Best Model