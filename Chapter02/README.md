
In chapter 01, we mentioned using RAPIDS for processing and analyzing data with a GPU. In this article, I will guide you on how to use it.

<p align="center">
 <h1 align="center">RAPIDS in handle data </h1>
</p>

<p align="center">
 <h2 align="center"> cuDF </h2>
</p>

**cuDF, also known as CUDA DataFrame**, is used for processing data in CUDA. One great thing is that the **syntax of cuDF is exactly the same as pandas**, but in terms of performance, **it's 10 to 400 times faster.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/0a03ed9e-a4b6-4634-ba03-e833c82ac24b" />
</p>

<p align="center">
 <h2 align="center"> How cuDF work </h2>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/3d429292-4a8a-4322-b979-dee7a57c97d9" />
</p>

You can simply understand that when you call **%load_ext cudf.pandas**, importing pandas (or any of its sub-modules) **will not actually work like a regular pandas library**. Instead, it will call a **proxy module (a replacement/temporary module) that temporarily substitutes the components in pandas**. This allows you to harness the **power of the GPU** for data processing while still maintaining the same pandas syntax.


<p align="center">
  <img src="https://github.com/user-attachments/assets/112718a5-6829-4142-b0c8-c967b65050a4" />
</p>


- First, when we **load cudf.pandas, it will "spoof" the pandas module**, meaning that when you **call pandas**, you are actually using **cudf.pandas.**
- Your code or third-party libraries that use pandas will all function through cudf.pandas.

**example**

  ```
%load_ext cudf.pandas

import pandas as pd  
import seaborn as sns

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})

df['C'] = df['A'] + df['B']
print(df)

sns.barplot(x='A', y='B', data=df)
  ```

**In this case, df['A'] + df['B'] will be executed on the GPU (via cudf.pandas), and if you use seaborn (as a third-party library that uses pandas), the plot will also be processed on the GPU.**


<p align="center">
 <h3 align="center"> Here are some third-party libraries that support cudf for GPU acceleration </h3>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/0a7cdf08-58ff-49c1-b1eb-e6a2614ac4e3" />
</p>

- When we call pandas functions, cudf.pandas will handle them **if there are corresponding functions in cudf.**

- If there are **no equivalent functions** in cudf, the data will be copied back to the CPU to use the original pandas version. After processing, it can be copied back to the GPU if needed or simply print the results.



<p align="center">
 <h1 align="center"> Code </h1>
</p>


<p align="center">
 <h2 align="center"> In Notebook </h2>
</p>

```
%load_ext cudf.pandas
import pandas as pd
```

```
%%cudf.pandas.profile
df = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 3]})

df.min(axis=1)
out = df.groupby('a').filter(
    lambda group: len(group) > 1
)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/b88fe629-d958-4b82-83b0-e881471c143f" />
</p>

```
%%cudf.pandas.line_profile
df = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 3]})

df.min(axis=1)
out = df.groupby('a').filter(
    lambda group: len(group) > 1
)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd2536f2-8368-488f-a54d-c287fe3af6c1" />
</p>

<p align="center">
 <h2 align="center"> In Local </h2>
</p>


```
import pandas as pd

df = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 3]})

df.min(axis=1)
out = df.groupby('a').filter(
    lambda group: len(group) > 1
)
```

**Run this command to enable cudf.pandas**

```
$python3 -m cudf.pandas <file name>.py
```

**-m cudf.pandas** work the same as **%load_ext cudf.pandas**

**If you want to profile, run this**

```
$python3 -m cudf.pandas --profile <file name>.py
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/f3201674-edad-42e3-9b88-9f2c8875ed24" />
</p>

```
$python3 -m cudf.pandas --line-profile <file name>.py
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/2cc73cd6-f67f-4d98-84bf-ca86c8d0b1fd" />
</p>

`From the demo code, we can see that DataFrameGroupBy.Filter is not supported on the GPU, so it is automatically executed on the CPU. In the next article, I will guide you on how to write custom kernels to support GPU processing to achieve maximum optimization.`

<p align="center">
 <h1 align="center"> Question </h1>
</p>

From those images, why is the total time not equal to the sum of the CPU and GPU times?














