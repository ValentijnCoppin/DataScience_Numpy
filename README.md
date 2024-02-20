The git repo for this project: https://github.com/ValentijnCoppin/DataScience_Numpy

# Creating the virtual environment
## install Poetry
```bash
pip install poetry
```

## adding a virtual environment to your project
If you want to create a virtual environment inside your project
This would make it much simpeler to delete everything in case you delete the project
Poetry will automatically install in the .venv forder if it exists
```bash
mkdir .venv
```


## installing dependencies
```bash
poetry install --no-root
```

## Activating the virtual environment
```bash
.venv\Scripts\activate
```

## add dependencies to the project
If you want to add a dependency to the project
```bash
poetry add <package-name>
```

Main notebook: notebooks/image_manipulation.ipynb

Functions used in the main notebook: scripts/numpy_image_manipulation.py
Picture used in this exercise: data/input/lala.png (this is not uploaded to GIT)