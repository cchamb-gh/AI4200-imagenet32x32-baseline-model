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

**Change:** double the number of training epochs for mnist but otherwise make no changes. Epochs went from 10 to 20.

**Why:** Validating HPC setup. I want to be sure I can submit this work as a batch job and run headlessly so I don't have to babysit it.

**Expected:** basic improvement resulting from just doing more training on MNIST

**Result:** mnist accuracy went from 95.06% to 97.23%. Training duration grew from 0.6 to 1.1 minutes.

```
2026-10-08 13:50,git+https://github.com/cchamb-gh/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,mnist,0.9723,1.1,0
```




### Change 2 - mnist and cifar - 2026-10-15

**Change:** switch to ReLU activations for the hidden blocks and update initialization to He Kaming. I'm going to use the Week 5.1
colab notebook that gives the LayerNormBlock template with ReLU and He Kaming initialization.

**Why:** in a future change, I will attempt adding depth to the network. ReLU performs better for deep networks, so this is an intermediate step.

**Expected:** I don't expect to see much change in accuracy. Since I'm not changing depth yet, I expect training time to decrease a little since ReLU activation is computationally cheap.

**Result:** There was a decrease in training time for cifar and imagenet from the baseline. cifar trained 30 epochs in 1.5 minutes, and imagenet trained 10 epochs in 13.7 minutes. Both of those have the original epoch count. I'm a little skeptical of accurate timing, though. The HPC node I was executing on had four other users with heavy loads executing at the same time. there may have been some resource competition. As for accuracy, I saw a very small improvement in accuracy for cifar (0.48 to 0.49) and imagenet (0.07 to 0.08).


### Change 3 - mnist - 2026-10-16

**Change:** continuing to add infrastructure by created a specific MNIST model. For the initial model, I'm going to try a simple FFN with depth of 13 and tanh activation. MNIST will use the new mnist_model.py file and its class. Other tasks will continue to use general model.py and its class. 

**Why:** trying to achive two things. First, add infrasturcture to support specializing the models for each task. MNIST will be specialized first, but the same can be done later for the other tasks. Second, see if a simple FFN can do as well as a residual MLP. The class notes from Week 7.2 have an example of an FFN with a single hidden layer getting 97.38% accuracy with 20 epochs.

**Expected:** I expect my MNIST-specific FFN with 13 hidden layers will improve upon the accuracy from the single hidden layer example from the Week 7.2 notes. It's a simple network.

**Result**: 
