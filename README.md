# HYBRID RECOMMENDATION SYSTEM USING BERT AND NEURAL NETWORKS FOR E-COMMERCE


## Usage

**Requirements**

* python 3.10+
* Tensorflow 2.15 (GPU version)
* CUDA compatible with TF 2.15

**Run**

We have take amazon fashion dataset which is available here:
https://amazon-reviews-2023.github.io/main.html#:~:text=Amazon_Fashion,510.5M


For the Exploratory Data Analysis the jupyter notebook file should be executed

```
Hybrid-Recommendation-Model.ipynb
```

For the Model execution with sentiment analysis, User-Item and Item-User Matrix factorization based Recommendation model and Combining the outputs of BERT models the python file should be executed

``` bash
python Hybrid-Recommendation-Model.py >run.log
```