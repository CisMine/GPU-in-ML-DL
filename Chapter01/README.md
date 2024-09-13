
Nowadays, the increasing amount of data has made ETL (Extract, Transform, Load), also known as data analysis and processing, more complex and time-consuming. To address this issue, **NVIDIA created RAPIDS**

<p align="center">
 <h1 align="center"> RAPIDS in GPU </h1>
</p>

When it comes to data analysis and processing, we mostly think about **Python, Pandas, SQL, Spark,...** However, all these languages have a **major drawback** is they run on the **CPU**, which leads to **slow data processing and inefficient use of computer resources. This is why RAPIDS was developed.**


<p align="center">
 <h2 align="center"> What is RAPIDS </h2>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/1de15858-b49c-4649-8f7c-21e576def967" />
</p>

RAPIDS is an open source software libraries and APIs give you the ability to execute end-to-end data science,analytics and machine learning pipelines entirely on GPU. 

One great thing is that the syntax is completely similar to Pandas, NumPy, scikit-learn, and others.

**In RAPIDS, the main libraries include:**
- cuDF: like pandas but run in GPU
- cuML: like Sklearn but run in GPU
- cuGraph: like NetworkX but run in GPU
- cuSpatial: like GIS but run in GPU


<p align="center">
  <img src="https://github.com/user-attachments/assets/d37e35c2-0512-481c-8aa0-b7a38a9bb966" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/331fcf05-5c42-42da-8e06-8252f161d427" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/1c900f33-bdc4-4106-8d3f-a8f33552bce2" />
</p>

**Through the above images, we can see that RAPIDS demonstrates a superior speed compared to other libraries, and the great thing is that the accuracy remains unchanged.**

`One note is that in the "GPU in AI" series, I will only guide on the two main libraries: cuDF (Pandas on GPU) and cuML (Sklearn on GPU).`



<p align="center">
 <h1 align="center"> Set up </h1>
</p>

<p align="center">
 <h3 align="center"> In Local </h3>
</p>

Here is the [link](https://docs.rapids.ai/install) to install RAPIDS

<p align="center">
  <img src="https://github.com/user-attachments/assets/97b7960a-c40a-4ee7-8ca9-a43f79618290" />
</p>

Please select as shown above, BUT remember to check which version of the **CUDA toolkit** you have by using the following command:

```
$nvcc -V
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/b9c6d8a6-e62e-487d-bc5c-db0cac93ccda" />
</p>

If you haven't installed the **CUDA toolkit** yet, you can refer to the guide [here](https://github.com/CisMine/Setup-as-Cuda-programmers/blob/main/Cuda%20Toolkit%20-%20Cuda%20Driver.md).

**One important note is that RAPIDS only supports Python versions 3.9, 3.10, and 3.11.**

After installation, you can verify by using the following commands:

```
import cudf
cudf.__version__

import cuml
cuml.__version__

import cugraph
cugraph.__version__

import cuspatial
cuspatial.__version__

import cuxfilter
cuxfilter.__version__
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/2865228a-df3e-4d8c-a91b-d7a89f61e3df" />
</p>

<p align="center">
 <h3 align="center"> In Google Colab </h3>
</p>

Change the runtime from CPU to GPU 

<p align="center">
  <img src="https://github.com/user-attachments/assets/63d6fe66-c089-43e1-9978-a26fc542815a" />
</p>

run these commands:
```
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git

!python rapidsai-csp-utils/colab/pip-install.py
```

After installation, you can verify by using the following commands:

```
import cudf
cudf.__version__

import cuml
cuml.__version__

import cugraph
cugraph.__version__

import cuspatial
cuspatial.__version__

import cuxfilter
cuxfilter.__version__
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/a01ddf97-4aaa-48ae-9890-265c5797eab8" />
</p>































