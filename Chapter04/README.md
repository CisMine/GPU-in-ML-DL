

<p align="center">
  <img src="https://github.com/user-attachments/assets/3edf1b38-caf4-4600-b995-7d650f3d71ec" />
</p>


This is a complete pipeline for developing a Machine Learning model. In this article, we will focus on the tasks within the red box, including **Feature Engineering, Model Selection, Model Training, and Hyperparameter Tuning.**

One of the biggest challenges is finding the optimal model with the most suitable hyperparameters for the problem. This process often requires a significant amount of time and effort, as multiple methods need to be tested to achieve the best performance.

Here, we have two well-known techniques:

- **Lazypredict:** It only identifies models with default parameters.
- **GridSearchCV / RandomSearchCV:** Helps find hyperparameters but requires manual execution.
  
➡ **Lazypredict + GridSearchCV:** Very slow and inaccurate.

- **Reasons for being slow:** After using Lazypredict to select top models, we then apply GridSearchCV to find hyperparameters for each model. This takes a lot of time because GridSearchCV uses an iterative process, meaning that the more hyperparameters we have, the longer it takes.

- **Reasons for inaccuracy:** We may mistakenly assume that the top models identified by Lazypredict will continue to perform well with optimized hyperparameters. However, some models that do not rank high with default parameters could actually perform better when combined with the right hyperparameters.

Therefore, in this article, I will guide you on how to find the best model with optimal hyperparameters for our problem while leveraging GPU acceleration to speed up processing.


<p align="center">
 <h1 align="center"> TPOT AutoML + cuDF</h1>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/b40fb65f-bfba-46c3-b7b8-edf9640b5d87" />
</p>


**TPOT (Tree-based Pipeline Optimization Tool)** is a tree-based pipeline optimization tool. As simple,**TPOT combines Lazypredict and GridSearchCV**, but instead of searching randomly or sequentially like GridSearchCV, TPOT uses **genetic programming** to optimize the model.

Genetic programming can be understood as a learning process based on inheritance and development from previous generations. To make it easier to visualize, let's consider a competition with multiple elimination rounds:

- In each round, weaker candidates are eliminated, while stronger candidates continue to the next stage.
- The more elimination rounds there are, the higher the quality of the selected candidates.
- However, if there are too many elimination rounds, there may be no candidates left who meet the criteria, causing the optimization process to fail.
  
This is exactly how genetic programming works in TPOT: it continuously improves the model by selecting, mutating, and combining the best models across multiple generations to find the optimal structure. **The entire process is automated.**


<p align="center">
  <img src="https://github.com/user-attachments/assets/069e6f23-2df7-4897-8f30-52bac856e302" />
</p>



<p align="center">
 <h1 align="center"> Code</h1>
</p>


`Here, I'm using the standard version of TPOT, so the data processing step needs to be done manually. However, if you use TPOT2 (alpha), it will handle data processing for you automatically.`

```
import cudf
from cuml.preprocessing import LabelEncoder
from tpot import TPOTClassifier

data = cudf.read_csv('/kaggle/input/1-3m-linkedin-jobs-and-skills-2024/linkedin_job_postings.csv')
data = data.dropna()

encoder = LabelEncoder()

for col in data:
    data[col] = encoder.fit_transform(data[col])

y = data["job_level"]
x = data.drop("job_level", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.75, test_size=0.25)

tpot=TPOTClassifier( generations=3,
   population_size=2,
   config_dict="TPOT cuML",
   memory='auto',
   scoring='roc_auc',
   max_time_mins=40,
   cv=2,
   verbosity=2)

tpot.fit(x_train.to_numpy(), y_train.to_numpy())
print(tpot.score(x_test.to_numpy(), y_test.to_numpy()))
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/f40f21cb-d42b-474f-9402-15d05847ce6c" />
</p>



With just a few simple steps, we've found the best model with optimal hyperparameters and integrated it with cuDF to run on GPU for acceleration. Here’s a brief explanation of the code:

**Important Note**
To use TPOT, we need to convert the data type to NumPy. However, the code will still run smoothly on GPU because cuML is flexible with data types, meaning NumPy arrays can still be processed efficiently.

**Key Parameters in TPOT:**
- **generations:** The number of elimination rounds in the "competition," meaning the number of generations the genetic algorithm will run to improve the model. The more generations, the better the optimization, but it also requires more computational resources.
- **population_size:** The number of models participating in the hyperparameter search in each generation. A larger population size gives the algorithm more choices, increasing the chance of evolving better models.


<p align="center">
 <h1 align="center">Warning in using cuDF for reading file</h1>
</p>


```
data_cpu = pd.read_csv('/kaggle/input/student-score/StudentScore.xls')
print(data_cpu.info())

data_gpu = cudf.read_csv('/kaggle/input/student-score/StudentScore.xls')
print(data_gpu.info())
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/4b9b1e45-9a7b-46ad-baec-7e8c01bd0e6c" />
</p>


You can see that **math score, reading score, writing score** are converted into object


<p align="center">
 <h1 align="center">Handle </h1>
</p>


1. Reading file in pandas then convert back to cuDF

```
data = pd.read_csv("/kaggle/input/student-score/StudentScore.xls")
data = cudf.DataFrame.from_pandas(data)
print(data.info())
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/af2eae7f-5abb-48aa-bc57-0dc234750f77" />
</p>


2. Point out specific data types when reading file in cuDF


```
dtype_mapping = {
    "Math Score": "int64",
    "Reading Score": "int64",
    "Writing Score": "int64"
}

data_gpu = cudf.read_csv('/kaggle/input/student-score/StudentScore.xls', dtype=dtype_mapping)

print(data_gpu.info())
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/dd048de6-6962-4e6d-ba77-b0578a4c9178" />
</p>












































