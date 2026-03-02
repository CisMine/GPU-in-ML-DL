

In this article, I will show you the power of Lightning Fabric through model initialization, compared to the conventional way in PyTorch.

<p align="center">
 <h1 align="center"> Optimized Model Initialization </h1>
</p>

**With the standard way of initializing a model in PyTorch, it looks like this**:

```
model = MyModel()
model.to(torch.device("cuda")).to(torch.float16)
```

There are two main issues here:

- **Speed**: The model is first initialized on the CPU. The larger the model, the longer it takes to copy it from CPU to GPU (due to bottlenecks). Additionally, when the model is created on the CPU, PyTorch uses float32 by default. Casting it to float16 afterward takes extra time.

- **Memory**: Since the model is initially created in float32, when it is copied to the GPU, the peak GPU memory will temporarily be in float32. For large models, this can easily lead to OOM (Out Of Memory) errors.

**Using Lightning Fabric**: 

```
fabric = Fabric(accelerator="cuda", precision="16-true")
fabric.launch()

with fabric.init_module():
    model = MyModel()
        
model = fabric.setup(model)
```

This approach solves both problems:

- **Speed**:The model is initialized directly on the GPU, without copying from CPU to GPU. It also uses half-precision from the beginning, so no extra casting step is required.

- **Memory**:Because the model is initialized directly in half-precision, the peak GPU memory usage is also half-precision.



<p align="center">
 <h3 align="center"> Demo </h3>
</p>

For this demo, I will use the **vit_l_16** model with the following benchmarks:

- **Time**: The time from model initialization until it is moved to the GPU.

- **Memory**: I will measure four metrics: Current Allocated - Current Reserved - Max Allocated - Max Reserved

`For now, you only need to focus on Current. I will explain these four metrics in detail in a later article about how GPU memory works in PyTorch.`

<p align="center">
 <h4 align="center"> Pytorch  </h4>
</p>




<p align="center">
  <img src="https://github.com/user-attachments/assets/46937057-906f-49e4-ada6-863de6ab5ece" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d672cd46-b416-4fa2-94ca-e698e6af4c43" />
</p>


<p align="center">
 <h4 align="center"> Lightning Fabric </h4>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/5e909b05-b29e-44d0-8513-a109c9a2a5bc" />
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/4945cba5-8194-4b78-8766-a7c8186a382e" />
</p>











































