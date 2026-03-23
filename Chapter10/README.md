

In this section, I will use mixed precision with Lightning Fabric as well as note some important considerations when using Fabric.

<p align="center">
 <h1 align="center"> Mixed Precision Part 2 </h1>
</p>

For traditional PyTorch, using mixed precision requires some changes:

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
For Fabric, however, everything is handled automatically, and we hardly need to modify the code during the model’s forward pass.

```
fabric = Fabric(accelerator="cuda", precision="16-mixed")  # 16-true
fabric.launch()

with fabric.init_module():
    model = MyModel()
        
model = fabric.setup(model)

... 

# Precision casting gets handled in your forward, no code changes required
output = model.forward(input)

# Precision does NOT get applied here (only in forward)
loss = loss_function(output, target)
```

This code snippet was already mentioned in [Chapter07](https://github.com/CisMine/GPU-in-ML-DL/tree/main/Chapter07), so I will skip the explanation here. We only need to change from 16-true to 16-mixed.


Since PyTorch already has a loss scaling mechanism to ensure stable transitions between float16 and float32, computing the loss with mixed precision or not has almost no impact on the results. However, if you still want to use float16 for the loss, here is the code:

```
with fabric.autocast():
    loss = loss_function(output, target)
```

There are others different precisions in Fabric:

```
# This is the default
fabric = Fabric(precision="32-true")

# Also FP32 (legacy)
fabric = Fabric(precision=32)

# FP32 as well (legacy)
fabric = Fabric(precision="32")

# Float16 mixed precision
fabric = Fabric(precision="16-mixed")

# Float16 true half precision
fabric = Fabric(precision="16-true")

# BFloat16 mixed precision (Volta GPUs and later)
fabric = Fabric(precision="bf16-mixed")

# BFloat16 true half precision (Volta GPUs and later)
fabric = Fabric(precision="bf16-true")

# 8-bit mixed precision via TransformerEngine (Hopper GPUs and later)
fabric = Fabric(precision="transformer-engine")

# Double precision
fabric = Fabric(precision="64-true")

# Or (legacy)
fabric = Fabric(precision="64")

# Or (legacy)
fabric = Fabric(precision=64)

```

**Note: bfloat16 and transformer-engine require newer GPUs, so if you force their usage on unsupported hardware, it may lead to slower performance or even errors.**


```
===== 16-mixed =====
Seed set to 42
Train Acc: 38.87%
Test Acc : 46.15%
Time     : 987.84s
Memory   : 5603.31 MB

===== bf16-mixed =====
Seed set to 42
Train Acc: 42.05%
Test Acc : 46.14%
Time     : 4992.44s
Memory   : 7415.20 MB

===== 32-true =====
Train Acc: 40.74%
Test Acc : 47.37%
Time     : 3692.28s
Memory   : 8805.79 MB
```

As you can see here, I used a T4 GPU on Kaggle, and the results show that bfloat16 performs far worse than float32 in terms of time, while memory usage is almost the same. The reason is that Kaggle’s T4 GPU does not support bfloat16.

Note that even if you test it using code like the following, it is still not reliable.


<p align="center">
  <img src="https://github.com/user-attachments/assets/849de9cb-e0e5-4464-8e2f-6705659c0e74" />
</p>


`I’ll cover how to determine whether a GPU is suitable in a separate post.`










































