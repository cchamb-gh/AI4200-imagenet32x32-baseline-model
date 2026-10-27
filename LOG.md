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


### Change 5 - mnist, cifar10, imagenet32 - 2026-10-19 - #519bc29e

**Change:** Specify an SGD optimizer momentum of 0.9 for all tasks.

**Why:** Optimizer momentum is supposed to help overcome cases where the gradient gets stuck in a
local minima, and it should give faster convergence

**Expected:** I expect the training loss to decrease faster from one epoch to the next. And accuracy
should improve for all tasks since it should step over any local minima that we were hitting
previously.

**Result:** 
Pracitically no change in MNIST. I thought it would be a slight improvement, but maybe we're already
close to optimum convergence and getting no benefit from momentum. CIFAR accuracy improved 5%, so
that's a nice, big jump. Bigger than I expected. Imagenet accuracy also increased but only by 1%.

```
2026-10-19 16:05,git+https://github.com/cchamb-gh/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,mnist,0.9705,0.3,0
2026-10-19 16:06,git+https://github.com/cchamb-gh/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,cifar10,0.5426,1.4,0
2026-10-19 16:19,git+https://github.com/cchamb-gh/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,imagenet32,0.0889,12.5,0
```


### Change 6 - mnist, cifar10, imagenet32 - 2026-10-19 - #688f7678

**Change:** Reduce microbatch size from 256 to 64

**Why:** Smaller microbatches will exercise backpropogation more often. Going from 256 to 64 will
give 4 times as many updates to weights and biases in every epoch.

**Expected:** Having more updates is probably, mostly going to be a good thing, but I worry that
doing more updates without adjusting learning rate may cause some jumping around as we approach
convergence. The default LR of 0.05 needs to be reduced overall or configured with a scheduler. I
predict some small improvement in accuracy of all tasks but perhaps some jumping around in the loss
in the later epochs. We may see loss grow from one epoch to the next. Training time will be slightly
longer since weights and biases are getting updated 4 times as often.

**Result:**
I need to look at loss rates in the notebook before drawing any conclusions here. Training definitely took longer for the cifar and
imagenet tasks. Much longer than I expected. Looks like mnist accuracy went down. Accuracy for cifar and imagenet both went up by
a percent. More to come

```
2026-10-19 16:56,git+https://github.com/cchamb-gh/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,mnist,0.8629,0.6,0
2026-10-19 17:00,git+https://github.com/cchamb-gh/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,cifar10,0.5574,4.8,0
2026-10-19 17:43,git+https://github.com/cchamb-gh/AI4200-imagenet32x32-baseline-model.git,residual_mlp_baseline,imagenet32,0.0924,42.5,0
```

*Updating with loss values*: I see loss begin rising around epoch 9 for mnist. It doesn't follow the desired trajectory and gets worse over time. I think this would be fixed with a smaller learning rate when coupled with smaller batches and optimizer momentum.

cifar and imagenet do show continued improvement as epochs increase, but it's not a fast improvement. They may need learning rate increased up to the point where we see oscillation in the loss rate.

```
===== mnist =====
epoch 1: train loss 0.4672
epoch 2: train loss 0.3132
epoch 3: train loss 0.2761
epoch 4: train loss 0.2668
epoch 5: train loss 0.2511
epoch 6: train loss 0.2475
epoch 7: train loss 0.2441
epoch 8: train loss 0.2472
epoch 9: train loss 0.2390
epoch 10: train loss 0.2687
epoch 11: train loss 0.2648
epoch 12: train loss 0.2672
epoch 13: train loss 0.2464
epoch 14: train loss 0.2576
epoch 15: train loss 0.3416
epoch 16: train loss 0.3700
epoch 17: train loss 0.3250
epoch 18: train loss 0.3346
epoch 19: train loss 0.4567
epoch 20: train loss 0.5048
mnist: test accuracy 86.29%  (training took 0.6 min)

===== cifar10 =====
epoch 1: train loss 1.7184
epoch 2: train loss 1.4807
epoch 3: train loss 1.3751
epoch 4: train loss 1.3016
epoch 5: train loss 1.2348
epoch 6: train loss 1.1791
epoch 7: train loss 1.1192
epoch 8: train loss 1.0685
epoch 9: train loss 1.0211
epoch 10: train loss 0.9694
epoch 11: train loss 0.9211
epoch 12: train loss 0.8707
epoch 13: train loss 0.8246
epoch 14: train loss 0.7736
epoch 15: train loss 0.7168
epoch 16: train loss 0.6761
epoch 17: train loss 0.6256
epoch 18: train loss 0.5839
epoch 19: train loss 0.5372
epoch 20: train loss 0.4943
epoch 21: train loss 0.4488
epoch 22: train loss 0.4154
epoch 23: train loss 0.3792
epoch 24: train loss 0.3428
epoch 25: train loss 0.3094
epoch 26: train loss 0.2813
epoch 27: train loss 0.2609
epoch 28: train loss 0.2325
epoch 29: train loss 0.2085
epoch 30: train loss 0.1937
cifar10: test accuracy 55.74%  (training took 4.8 min)

===== imagenet32 =====
epoch 1: train loss 6.0835
epoch 2: train loss 5.6758
epoch 3: train loss 5.4879
epoch 4: train loss 5.3552
epoch 5: train loss 5.2491
epoch 6: train loss 5.1568
epoch 7: train loss 5.0768
epoch 8: train loss 5.0050
epoch 9: train loss 4.9390
epoch 10: train loss 4.8782
imagenet32: test accuracy 9.24%  (training took 42.5 min)
```


### Change 7 - Telemetry and MNIST - 2026-10-22 - #

**Changes:** Restore MNIST to a residual MLP network and then add measurements for the following:

1. forward variance of the residual stream at every depth

2. correlation across samples at every depth (whether different inputs are still represented
   differently or are collapsing onto the same direction)

3. gradient RMS at every depth: the size of the backward signal reaching each block

This follows the Week 5.1 notebook shared by Dr. Roberts in colab.


**Why:** I need to be able to see what's happening inside the network to guide my future changes.
Need to cultivate an understanding of what's going right and wrong.

**Expected:** I'm only making changes to the MNIST model, and that's just to restore it to a basic
residual MLP. There may be some incidental change in performance from changing network types, but
the primary goal is to get the telemetry instrumentation up and running.

**Result:**

### Change 8 - diagnostics - 2026-10-25 - #05b7b6d29

**Change:** Get diagnostics and ablation study code working on HPC with imagenet32

**Why:** to help steer model and training regimen adjustments 

**Expected:** get some stats and graphs

**Result:** it works! needs some improvment, but it's working at a basic level. Gemini wrote the
telemetry and plotting scripts. I integrated them with the notebook and got it running interactively
on HPC.




### Change 9 - SwiGLU - 2026-10-26 - #db001ab2f

**Change:** Replace our normal residual MLP with SwiGLU

**Why:** We learned about SwiGLU and other gated activation functions in class in lecture 8.1. It
has the advantages of ReLU plus the additional ability to differentiate values less than zero. It
supposed to help with overfitting.

**Expected:** I should be able to build larger networks with less risk for overfitting. I saw some
overfitting in my previous mnist tests. I expect I should be able to push mnist further and get
better accuracy among the test dataset. That same advantage should apply to cifar10 and imagenet32,
although I don't think I've witnessed overfitting there yet.

**Result:** Awful! I must have a bug in my network. Test accuracy for mnist dropped to 11%, which is
just barely above random guessing. Clearly something went wrong. commenceing troubleshooting.

After doing some debugging, I see the input stream vanish and go entirely to zero after the fourth
block. Both mean and variance become zero by block 5.

### Change 10 - SwiGLU, bug fix - 2026-10-26 - #09b5376eb

**Change:** Fix my earlier bug in my SwiGLU implementation. I'd failed to add the residual stream
back to the output of the block. 

**Why:** Fixing obvious bug

**Expected:** Models should at least learn and do better than random guesses.

**Result:** Models did learn. The bugfix worked. Accuracy measurement was slightly lower than the
original ResidualMLP network from previous commits BUT training was much faster because the new
SwiGLU network was using the original batch size of 256. Training was 14.8 minutes compared to 42+
minutes for the 64 batch size. 

    * mnist:      97.75%,  0.7 minutes
    * cifar10:    51.29%,  1.6 minutes
    * imagenet32:  5.86%, 14.8 minutes

### Change 11 - He initialization for SwiGLU network - 2026-10-26 - #967cf0d0

**Change:** Restore the He initialization of the SwiGLU network blocks. I had just been allowing the
blocks' weights to be implicitly initialized, but we can be more deliberate to get better model
performance.

**Why:** Need to maintain variance throughout the depth of the network.

**Expected:** Some small improvement in accuracy with no impact to training time. Network should
continue to learn along its depth.

**Result:** It's worse! I didn't expect that. Cifar10 and imagenet32 both suffered decreased
accuracy on the test set. Perhaps I'm misunderstanding the right way to apply He initialization to
the SwiGLU network.


### Change 12 - input standardization - 2026-10-26 - #6bba0b84

**Change:** Apply standardization to input tensors.

**Why:** Gets the input variance close to 1, which is what our He-initialized layers are expecting.

**Expected:** Some improvement in accuracy over the non-standardized inputs.

**Result:** mnist increased by less than one percent going from 97.28% to 97.52%. cifar10 went from
44.73% to 45.05%. So, that's some improvment but really not very much. After making this change, I
saw documentation about standardizing invidiaul channels in the RGB images instead of just having a
single pixel mean and std. That makes sense. I'll put that into a future change.

Also note that I had an out-of-memory crash on imagenet32, so that didn't complete. I suspect it's
an issue with calculating the pixel mean and std for the whole dataset at once. I'll request more
memory in my job spec, and if that doesn't work, I'll remove the code that calculates those values
dynamically.


### Change 13 - adding accuracy and loss curves - 2026-10-26 - #203d56f8

**Change:** Add a way to track accuracy and loss during training by epoch so I can watch for issues.

**Why:** I suspect there's some overtraining based on some local tests I'm running with MNIST and
some intuition from previous runs. There may be overfitting with the other models as well.

**Expected:** I expect to see some graphs showing whether my models are suffering from underfitting
or overfitting. I suspect overfitting is more likely.

**Result:** Found massive (!!) overfitting on cifar10. Training accuracy was near 100%, but test set
accuracy was only around 47%. I'll write up more detail in my analysis report.


### Change 14 - address overfitting - 2026-10-27 - #

**Change:** Add image augmentation blocks to training. 

**Why:** We learned in class in week 8.1 that large networks are quite capable of memorizing the
training data, and that's certainly what appears to be happening with my cifar10 network. I imagine
it could also be happening with mnist and imagenet32 but it's not as pronounced. To address that, we
talked in class about image augmentation. Pytorch has some built-in image transformations available.
I searched for the most common transformations and looked at sample code to come up with a simple
set of augmentations suitable for our images. Starting simple, I'm hoping to add complexity later.
For now, we'll just do very basic **AutoAugment** for cifar10 and imagenet32.

**Expected:** I expect to see much less overfitting in the 3-channel datasets. That should improve
test accuracy and close the gap between train accuracy and test accuracy.

**Result:**


----- Template ------
### Change X - X - 2026-10-XX - #
**Change:**
**Why:**
**Expected:**
**Result:**

