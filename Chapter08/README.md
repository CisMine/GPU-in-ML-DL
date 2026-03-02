


In this article, I will discuss two important concepts in PyTorch GPU memory: Allocated and Reserved.

<p align="center">
 <h1 align="center"> Allocated - Reserved </h1>
</p>



<p align="center">
  <img src="https://github.com/user-attachments/assets/7bfce275-9708-4cb6-ab04-36835227d47c" />
</p>


**Allocated memory**: the amount of GPU memory that is actually being used by tensors at the current moment.

```
Create a tensor → Allocated increases

Delete a tensor → Allocated decreases
```

You can check it with: `torch.cuda.memory_allocated()`

**Reserved memory**: the total amount of GPU memory that PyTorch has requested from the CUDA API (backend: cudaMallocAsync).

Reserved includes:
- Allocated memory (currently used by tensors)
- Cached memory — memory blocks that were previously used but are not currently held by any tensor

`The purpose of caching is to reuse these memory blocks in future allocations instead of calling the CUDA API again. Therefore, when a tensor is deleted, its memory is not immediately returned to CUDA. Instead, it is moved to the cached state for future reuse.`

You can check it with: `torch.cuda.memory_reserved()`

**Other**: GPU memory that is not managed by PyTorch. This includes:

- CUDA context (created when CUDA is initialized)
- Backend libraries such as cuDNN, cuBLAS, NCCL
- Other processes using the GPU

Since this memory is not under PyTorch’s control:
- It cannot be freed using del or torch.cuda.empty_cache()
- It still appears in nvidia-smi



<p align="center">
 <h3 align="center"> Demo  </h3>
</p>



<p align="center">
  <img src="https://github.com/user-attachments/assets/9ce7d3b1-2b9c-49df-8dd1-cf29ab26ca1c" />
</p>

You guys can swap del and empty_cache() to know more clearly how memory work


<p align="center">
  <img src="https://github.com/user-attachments/assets/622c9645-031d-4d3f-a666-e48cd852f473" />
</p>





<p align="center">
  <img src="https://github.com/user-attachments/assets/78719e1f-1183-4dd9-a31d-03365711509d" />
</p>

Here you can see that:
- after cleanup = after matmul − before matmul
- The important point is that even if we change before matmul, the value of after cleanup remains the same.

This happens because of Other memory (specifically, cuBLAS in this case). As mentioned earlier, memory in the Other category is not managed by PyTorch. Therefore, it is not released by del or torch.cuda.empty_cache(), and it still appears in nvidia-smi.


<p align="center">
  <img src="https://github.com/user-attachments/assets/20d476f5-cb1f-4ca4-ae0d-33baa109d544" />
</p>

`Here, the reason is that Other memory is not under PyTorch’s control so the GPU memory reported by PyTorch and the memory shown in nvidia-smi will be different.`

You can explore this topic in more detail [here](https://discuss.pytorch.org/t/torch-cuda-memory-snapshot-shows-an-unexpected-8519680b-memory-allocation-during-matrix-multiplication/178671).











