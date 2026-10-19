### Change 0 - baseline run with no changes - 2026-10-06

**Change:** None

**Why:** Initial run in class

**Expected:** baseline performance

**Result:**


```
2026-10-06 16:16,git+https://github.com/JesseTNRoberts/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,mnist,0.9506,0.6,0
2026-10-06 16:17,git+https://github.com/JesseTNRoberts/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,cifar10,0.4861,1.3,0
2026-10-06 16:30,git+https://github.com/JesseTNRoberts/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,imagenet32,0.0677,12.4,0
```


### Change 1 - mnist - 2026-10-08

**Change:** double the number of training epochs for mnist but otherwise make no changes. Epochs
went from 10 to 20.

**Why:** Validating HPC setup. I want to be sure I can submit this work as a batch job and run
headlessly so I don't have to babysit it.

**Expected:** basic improvement resulting from just doing more training on MNIST

**Result:** mnist accuracy went from 95.06% to 97.23%. Training duration grew from 0.6 to 1.1
minutes.

```
2026-10-08 13:50,git+https://github.com/cchamb-gh/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,mnist,0.9723,1.1,0
```




### Change 2 - mnist and cifar - 2026-10-15

**Change:** switch to ReLU activations for the hidden blocks and update initialization to He Kaming.
I'm going to use the Week 5.1 colab notebook that gives the LayerNormBlock template with ReLU and He
Kaming initialization.

**Why:** in a future change, I will attempt adding depth to the network. ReLU performs better for
deep networks, so this is an intermediate step.

**Expected:** I don't expect to see much change in accuracy. Since I'm not changing depth yet, I
expect training time to decrease a little since ReLU activation is computationally cheap.

**Result:** There was a decrease in training time for cifar and imagenet from the baseline. cifar
trained 30 epochs in 1.5 minutes, and imagenet trained 10 epochs in 13.7 minutes. Both of those have
the original epoch count. I'm a little skeptical of accurate timing, though. The HPC node I was
executing on had four other users with heavy loads executing at the same time. there may have been
some resource competition. As for accuracy, I saw a very small improvement in accuracy for cifar
(0.48 to 0.49) and imagenet (0.07 to 0.08).


### Change 3 - mnist - 2026-10-16 - #b7af8dc7e

**Change:** continuing to add infrastructure by created a specific MNIST model. For the initial
model, I'm going to try a simple FFN with depth of 13 and tanh activation. MNIST will use the new
mnist_model.py file and its class. Other tasks will continue to use general model.py and its class. 

**Why:** trying to achive two things. First, add infrasturcture to support specializing the models
for each task. MNIST will be specialized first, but the same can be done later for the other tasks.
Second, see if a simple FFN can do as well as a residual MLP. The class notes from Week 7.2 have an
example of an FFN with a single hidden layer getting 97.38% accuracy with 20 epochs.

**Expected:** I expect my MNIST-specific FFN with 13 hidden layers will improve upon the accuracy
from the single hidden layer example from the Week 7.2 notes. It's a simple network.

**Result**: The 13-layer network very slightly improved upon the 1-layer network from the notes.
going from 97.38% to 97.63% accuracy. Basically a wash at the cost of additional training time and
code complexity. Not worth the effort. But it was successful in testing the MNIST-specific model so
that I can continue running with task-specific models on the HPC.


### Change 4 - mnist - 2026-10-19 #363ab7dd

**Change:** Fix a bug with my simple FFN MLP for MNIST. It occurred to me that my layers had an
additional layer leftover from when I adapted them from residual MLP to feed forward MLP. Each layer
was linear -> tanh -> linear, but it should have just been linear -> tanh. Also, add more
infrastructure support for task specific configuration. Now the train function will allow different
values for each task for learning rate, batch size, etc.

**Why:** Just fixing the bug so each of the hidden layers is just linear -> tanh. And making the
project support task-specific params.

**Expected:** The bug was causing me to have twice as many linear layers as requested, which would
have make any deep-network problems show up earlier. Depth 10 would have actually had 20 linear
layers, for example. I don't expect a big change in loss or accuracy.

**Result:** Training time decreased in half. It was already very quick since this uses simple
layers, but it went from 0.6 to 0.3 minutes of training time. The bug fix removed 13 layers, so we
just had a total of 13 hidden layers remaining. Accuracy dropped less than a percent to 97.04% for
MNIST. That tells me that the extra layers really weren't contributing much. 2026-10-19
15:20,git+https://github.com/cchamb-gh/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,mnist,0.9704,0.3,0


### Change 5 - mnist, cifar10, imagenet32 - 2026-10-19 - #

**Change:** Specify an SGD optimizer momentum of 0.9 for all tasks.

**Why:** Optimizer momentum is supposed to help overcome cases where the gradient gets stuck in a
local minima, and it should give faster convergence

**Expected:** I expect the training loss to decrease faster from one epoch to the next. And accuracy
should improve for all tasks since it should step over any local minima that we were hitting
previously.

**Result:**



----- Template ------
### Change X - X - 2026-10-XX - #
**Change:** **Why:** **Expected:** **Result:**
