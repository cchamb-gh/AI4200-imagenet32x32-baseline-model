# Training Dynamics Analysis #


## Baseline run after adding telemetry and analytics scripts ##
* Running imagenet32 with 60,000 test images.
* commit 05b7b6d29a
* Model: ResidualMLP - linear -> ReLU -> linear. output activation: sigmoid
    * Width: 128
    * Hidden: 256
    * Depth: 30
* Training Regimen
    * Epochs: 5
    * batch size: 256
    * learning rate: 0.05
    * optimizer momentum: 0.9
* Accuracy: 1.12%

### Notes ###
First run after adding telemetry script. Just used 60K images for starters.

I see the residual stream growing at the expected rate as a function of depth.

Branch to stream ratio decreases and gets close to zero in later layers.

![imagenet32 output 1a](./graphs/output_1a.png)
![imagenet32 output 1b](./graphs/output_1b.png)

