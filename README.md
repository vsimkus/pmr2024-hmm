# PMR computer tutorials on HMMs (2021-2022)

This is a repository for computer tutorials of [Probabilistic Modelling and Reasoning (2021/2022)](http://www.inf.ed.ac.uk/teaching/courses/pmr) - a University of Edinburgh master's course.

The tutorial consists of three parts:

1. [HMM basics](<./HMM basics.ipynb>)
2. HMM inference (to be released)
3. HMM learning (to be released)

## Environment setup

Before you start with the tutorials you will first need to setup the environment on your preferred machine. The tutorials will use simple examples, hence any machine will do.

### Setup on your machine

You'll need to open terminal on your machine and then follow the below instructions

* Install git ([linux](https://git-scm.com/download/linux), [macOS](https://git-scm.com/download/mac), [windows](https://git-scm.com/download/win)) to access the repository if you don't have it already
* Clone the git repository on your machine by running `git clone` in the terminal (you can find a guide [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository))
* Once you've cloned the repository, step into the directory by entering `cd pmr2022-hmm` into the terminal
* If you don’t already have it also install miniconda  ([linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html), [macOS](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html), [windows](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)), which will allow you to manage all python dependencies per project
* You can now create the `pmr` conda environment by typing `conda env create -f environment.yml`. This step may take a while to complete since it has to download large binaries and you should better be connected to a good internet connection.

#### Starting the Jupyter server

Once you have the environment prepared you can start your jupyter notebook

* Activate the conda environment with `conda activate pmr`
* Now you will be able to start your jupyter server by typing `jupyter notebook`, which will start the server and open a browser to access the tutorial notebook. Click tutorial link in the browser window. You can stop the server by pressing <kbd>Ctrl</kbd>+<kbd>c</kbd> (or <kbd>Cmd</kbd>+<kbd>c</kbd>) in the terminal when you are done with it.

### Google Colab

You can also access and run the notebooks on Google Colab directly via this link <http://colab.research.google.com/github/vsimkus/pmr2022-hmm>. More details can be found at <https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=WzIRIt9d2huC>.

Note that the Colab notebook environment should already include all the required dependencies, however, the versions may differ, hence the results may differ slightly from the provided solutions but that should not be a problem for this tutorial.

## Attributions

The tutorials in this repository were authored by [Yao Fu](https://github.com/FranxYao/) and [Shangmin Guo](https://github.com/Shawn-Guo-CN) in discussion with [Michael Gutmann](https://michaelgutmann.github.io/), and edited by [Vaidotas Šimkus](https://github.com/vsimkus).
