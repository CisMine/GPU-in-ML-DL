

In this article, I will introduce a technique that helps reduce a model’s memory usage and speed up the training process, while keeping the accuracy almost unchanged.

`In PyTorch, when tensors or model parameters are initialized, the default data type is usually float32. This means that the training process is performed using 32-bit precision operations. Although float32 provides high numerical precision, it also causes the model to consume more memory and increases computation time, especially for large models.`

The core idea here is to scale the computation from float32 down to float16, which helps:
- Significantly reduce the required memory usage
- Increase computation speed, especially on GPUs that support Tensor Cores

<p align="center">
  <img src="https://images.viblo.asia/71776faa-9422-46e9-92b9-f2b04db3b8ea.png" />
</p>


`A simple way to understand this is:`

`- CUDA cores are used for general GPU computations (such as +, −, ×, ÷ operations).`

`- Tensor cores are specialized cores designed for AI workloads, optimized for operations like matrix multiplication and convolution.`

However, using float16 also means that numerical precision may drop significantly. To address this issue, mixed precision was introduced.



<p align="center">
 <h1 align="center"> Mixed Precision Part 1 </h1>
</p>

**Mixed Precision** is a technique that combines two data types, float16 and float32, during the training process. In this approach: **Large computational operations** (such as convolution or matrix multiplication) are performed in float16 to reduce memory usage and accelerate computation. Critical components, such as **loss calculation and weight updates**, remain in float32 to maintain model stability and numerical accuracy.

<p align="center">
  <img src="https://images.viblo.asia/6721274e-9222-4a2c-bfb4-a1d71b8d8385.png" />
</p>


The training process typically works as follows:
- The model is initialized (by default in float32).
- During the forward pass, computations are converted to float16.
- The loss is still computed in float32 to preserve precision.
- The loss is scaled up (multiplied by a scaling factor) to prevent it from becoming too small, which could cause gradients to underflow when using float16.
- After the backward pass in float16, the gradients are converted back to float32 in preparation for the weight update step.
- The gradients are then divided by the scaling factor applied earlier to restore their original magnitude.
- Finally, the FP32 gradients are used to update the FP32 master weights through the optimizer. After that, the training cycle repeats for the next batch.



<p align="center">
 <h3 align="center"> Code </h3>
</p>

```
from torch.amp import autocast, GradScaler

scaler = GradScaler()

with autocast("cuda"):   
    output = model(x)
    loss = criterion(output,y)

scaler.scale(loss).backward()  
scaler.step(optimizer)         
scaler.update()   
```

Here, autocast determines which operations should use float16 and which should remain in float32. Meanwhile, the scaler handles the problem of very small gradients (underflow) when computing with float16 by scaling the loss before the backward pass.

You can try printing the loss before and after scaling, as well as the data type of the output, to better understand how this mechanism works.

<p align="center">
 <h3 align="center"> Demo </h3>
</p>

In this demo, we will train a CIFAR-10 model using ResNet-18 for 2 epochs to compare training with and without mixed precision, and observe the results achieved.

```
===== RESULTS =====
FP32 Accuracy: 0.7114
FP32 Memory : 3219.65 MB
FP32 Time   : 309.41 sec

Mixed Accuracy: 0.6969
Mixed Memory  : 1916.57 MB
Mixed Time    : 252.56 sec

```






















