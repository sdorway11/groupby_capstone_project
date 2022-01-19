# groupby_capstone_project

I use pyenv for installing the python modules and poetry. 
if you are interested here is how.

here is a good site for installing pyenv and pyenv-virtualenv https://akrabat.com/creating-virtual-environments-with-pyenv/

after that is installed following those instructuctions run this in the directory of this project

```shell
pyenv install 3.9.9
pyenv virtualenv 3.9.9 groupby_capstone
pyenv local groupby_capstone
python -m pip install poetry
python -m poetry install
```

To add any modules run
```shell
python -m poetry add <module name>
```
exmaple:
```shell
python -m poetry add pandas
```

That should help maintain the same versions of the modules and python