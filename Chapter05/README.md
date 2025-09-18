One of the issues when we use GPUs is OOM (out of memory). So, in this lesson, I’ll show you a tip to avoid this problem.


<p align="center">
 <h1 align="center"></h1>
</p>

<p align="center">
 <h1 align="center">Parquet format for ML</h1>
</p>


<p align="center">
 <h3 align="center">Principle of memory operation</h3>
</p>

Before we dive into the lesson, we need to understand how memory works—how it stores and accesses data.


<p align="center">
  <img src="https://github.com/user-attachments/assets/5141a9dc-5f0c-4718-aaeb-f1b216b13990" />
</p>


From a coder’s perspective (also called the logical view), our data can exist in forms like 1D, 2D, 3D, and so on. But for the computer, it’s just one long continuous array.

For example: int a[3][3]. We see it as a matrix with 3 rows and 3 columns, but the computer stores it as one long array a[9].

There are three common ways for a computer to store multidimensional data:
- Row-wise: data is stored row by row.
- Columnar: data is stored column by column.
- Hybrid: data is not stored entirely by rows or columns, but split into smaller blocks.

`Here, I won’t go deep into Hybrid, but will focus on row and column storage.`


<p align="center">
 <h3 align="center">Which method is best for ML?</h3>
</p>


In machine learning tasks, the data we usually deal with comes in tabular form. When processing, we often work column by column (based on features). That’s why columnar storage is more suitable for ML tasks than row-wise storage.

Here, we’ll dive deeper into why columnar is the better choice.

<p align="center">
 <h5 align="center">Data sparsity</h5>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/91fe8277-bf38-4aaf-8f6e-14254485904d" />
</p>

In the illustration, there are two ways of storing data (red and black).
- In red, the data is stored continuously.
- In black, the data is stored with gaps.

The problem arises when we need to access red/black data:
- If the data is stored continuously → the CPU/GPU can load an entire chunk into cache, making it very efficient.
- If the data is stored with gaps, even though we only need a few elements, the CPU/GPU is forced to load entire chunks containing them. That means instead of just fetching 4 values, memory has to pull in a lot of unnecessary data → wasting bandwidth and cache.
- If the data is scattered randomly → memory has to perform fragmented accesses, resulting in very poor performance.

Back to our problem: if we store data in column format, processing features (columns) becomes much more efficient than storing in row format.


<p align="center">
 <h5 align="center">CSV is no longer suitable</h5>
</p>

CSV stores data row by row, so it doesn’t fit well with ML tasks. Instead, we should use Parquet — a file format that stores data in columnar format. Besides columnar storage, Parquet also supports smarter compression and encoding mechanisms, and Parquet files are often much smaller than CSV.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3de4b00b-a127-45ab-95dd-a808906ba699" />
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/f790f73f-8dd0-4c73-a89d-4efd8fffa7e8" />
</p>


As you can see in the figure, just by changing the file format (without altering any information), the file size has been reduced significantly — with only a single line of code.

```
import pandas as pd

df = pd.read_csv("data.csv")

df.to_parquet("data.parquet")
```

We will apply Parquet into GPU


<p align="center">
  <img src="https://github.com/user-attachments/assets/c4f7f150-c333-4937-97a9-3b59f59187dc" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/240c34dc-4d4a-4f47-9563-897976d21228" />
</p>

