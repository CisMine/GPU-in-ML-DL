

<p align="center">
 <h1 align="center">cuML for Machine Learning</h1>
</p>
 

In the previous article, we explored how to use the GPU to process data with cuDF. In this article, I'll guide you to use **cuML to handle machine learning tasks in GPU.**

`Before diving into today's article, let's answer a question from the previous one: Why is the total time not equal to the combined time of CPU and GPU ? --> This is because profiling does not account for the data copy time between the CPU and GPU.`

<p align="center">
 <h1 align="center">cuML</h1>
</p>


As I mentioned before, cuML is one of the libraries within RAPIDS, designed to help us easily use ML models in GPU with a syntax almost the same of sklearn.

<p align="center">
 <h1 align="center">Code</h1>
</p>


Let's first analyze the basic data processing steps.

<p align="center">
  <img src="https://github.com/user-attachments/assets/90ae2479-42d7-4c7e-beb1-dbc93f820228" />
</p>

Here we can see that the entire dataset is made up of objects, with some columns having missing values.

**Now it's time to compare the speed between the CPU and GPU when processing data, building the model, and making predictions.**


<p align="center">
 <h1 align="center">CPU</h1>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d364a30b-3896-40d6-978e-d0fc55464b41" />
</p>



<p align="center">
 <h1 align="center">GPU</h1>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/879a07f0-6f96-4528-86b4-6367c74e0752" />
</p>


As you can see, we made almost no changes to the code (just a small change in the library import: sklearn → cuml), but the **processing speed increased by 26 times.** However, the **accuracy dropped slightly** because when using cuML, our data type is **mapped to uint8, uint16, or uint32 to improve processing speed and reduce memory usage.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/6194ba21-4ff5-41a6-89a5-8590d47f2503" />
</p>


**One thing to note is that cuML doesn’t simply convert data to uint8, uint16, or uint32; it applies a technique called quantization — a technique that helps reduce the bit size when storing data, while keeping the changes in results minimal.**

```
For example, at the beginning, we needed a memory usage of 144.0+ MB, but with cuML, the memory usage dropped to 47.6 MB.

In machine learning tasks, I won’t go too deep into the quantization technique, but I will explain it more thoroughly in later articles when we apply GPU for deep learning.
```

From this, we can see that the GPU’s processing speed is extremely powerful and easy to use. **However, it’s not always necessary to use the GPU**, because:

To use the GPU, our data needs to be copied from the CPU (also known as the host) to the GPU (the device). So, if our dataset is too small, the copy time between the CPU and GPU could become a major bottleneck. Therefore, **we should only consider using the GPU for large datasets.**

```
What is considered a large dataset?
Since each computer has different processing speeds, a dataset might be considered large for one person but small for another. Here are two ways to determine if the dataset is large enough to justify using the GPU:

Simply read the CSV file using both CPU and GPU, then compare the results. If there’s a significant difference, use the GPU; otherwise, stick to the CPU.

Analyze the code using NVIDIA’s tools to determine whether using the GPU is appropriate. If you don’t know how to use NVIDIA tools, you can check out at [here](https://github.com/CisMine/Guide-NVIDIA-Tools/)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa448ad0-d204-42e2-81aa-8029ac8bb194" />
</p>


The data will be [here](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024)
