# online-x
A bunch of online algorithms for matrix factorization and collaborative filtering (**Online** algorithm mi**X** for machine learning).

### Overview
This repo provides implementation of `Passive-Aggresive` `Gibbs sampling LDA` `Online Collaborative Topic Regression`, see below for deatils.

### Quick guide
1. install `cmake`
2. navigate to the root of the source tree
3. `mkdir build && cd build && cmake ..`
4. if `cmake` throw no error, then `make`.
5. modify `run/call.sh` which will invoke the program.

### Parameter setting
Every parameter is fed into the program in a fixed position. If the parameter is not used in a specific algorithm, it will be ignored. This package only deal with diagonal value of a covariance matrix.

_synposis_: `./cppWrapper $algo $K $U $V $T $I $J $Jburnin $e $c $a0 $b0 $lambda_u $lambda_v $sigma_u $sigma_v $train_file $test_file $learn_cnt $test_cnt $basedir $test_interval $cdk_file $ofm_method`


`$algo`: this parameter is to set the algorithm to be run. Possible values are `ostmr`(obi-ctr) `osgd-topic-flow`(odi-ctr) `osgd-topic-fixed`(bdi-ctr) `gibbs-lda` `pa-i`. For difference of these methods please see [our paper at arXiv](https://arxiv.org/abs/1605.08872).

`K`: length of the latent variable or the length of the feature vector.

`U`: number of users.

`V`: number of items.

`T`: number of vocabulary in text corpus.

`I`: reserved for limiting iterations.

`J`: number of iterations for each Jibbs round (including burn-in samples)

`JB`: number of burn-in samples for each Jibbs round

`e`: reserved for limiting errors

`c`: only used when `algo` is `pa-i`. The 'step-size' in Passive-Aggresive.

`a0`: $\alpha_0$, hyper-parameter for word prior distribution.

`b0`: $\beta_0$, hyper-parameter for topic prior distribution.

`lu` `lv`: regularization $\lambda_v$, $\lambda_u$ for CTR (`bdi-ctr`, `obi-ctr`). Or $\sigma_r$ (please set these two params as identical values) for `obi-ctr`.

`su` `sv`: $\sigma_u$ $\sigma_v$ for `obi-ctr`.

`train` `test`: training and test filename.

`learn_cnt` `train_cnt`: number of tranining and test samples.

`base_dir`: base directory path for the above two files.

`test_interval`: test performance every `test_interval` samples seen.

`cdk_file`: import pretrained LDA model result for `bdi-ctr`.

### Data format

All IDs count from zero.

`train` `test` files should contain each sample in one line seperated by one space: `userid` `itemid` `score`.

A file named `_docs` file should exist, each line contains a list of `wordid` represents each word in the corresponding document. If a word appear several times, repeat its `wordid`.

License (GPL V3)
================

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
