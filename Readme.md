
# GPU-Accelerated Genetic Programming for Diabetes Classification
This repository features a high-performance Genetic Programming (GP) implementation developed using the DEAP framework and optimized for GPUs via CuPy. The project focuses on evolving symbolic mathematical expressions to solve binary classification tasks on large-scale datasets.

## 🚀 Key Engineering Highlights

**GPU Vectorized Evaluation:** Replaced traditional row-by-row Python loops with vectorized GPU operations. By transposing the feature matrix in VRAM, the evolved trees process all 95,804 rows of data simultaneously as a single parallelized operation.

**Bloat Control & Stability:** Implemented a dual-constraint system to prevent "survival of the fattest." Trees are capped at a height of 17 and a total size of 150 nodes, ensuring the generated Python expressions stay within the 200-nested-parenthesis limit of the Python parser.

**Memory Management:** Specifically optimized for VRAM hardware by integrating manual cache clearing for both system RAM and GPU memory pools between evolutionary runs.

**Evolutionary Strategy:** Utilizes an elitism-based approach (ea_with_elitism), preserving the top 2 individuals per generation to guarantee that the maximum fitness never regresses.

## 📊 Dataset: Diabetes Health Indicators (CDC BRFSS)
The model is trained on a dataset containing health-related survey responses from 95,804 individuals. The objective is to predict whether an individual has diabetes based on 21 clinical and lifestyle risk factors.

Target Variable
output: Binary classification (0 = No Diabetes, 1 = Diabetes/Pre-diabetes).

Key Feature Categories (21 Total)
To solve this, the Genetic Programming algorithm evolved a symbolic expression using features across three critical domains:

Clinical Indicators: **HighBP** (Blood Pressure), **HighChol** (Cholesterol), **BMI** (Body Mass Index), **GenHlth** (General Health rating).

Lifestyle Factors: Smoker, HvyAlcoholConsump, PhysActivity, Fruits, Veggies.

Socioeconomic/Demographic: Age, Sex, Education, Income, AnyHealthcare.

## Methodology
1. **Massive Data Parallelism (GPU Vectorization)**
Evaluating a population of 1,000 individuals across 95,000 rows per generation is computationally expensive.

*Solution:* I moved the entire dataset to VRAM and transposed the feature matrix. This allowed the evolved mathematical trees to be evaluated as a single vectorized operation on the GTX 1650, processing all rows simultaneously.

2. **Overcoming Python's Nesting Limits (Bloat Control)**
Genetic trees naturally "bloat," creating equations so deeply nested they exceed Python's 200-parenthesis limit and crash the parser.

*Solution:* I implemented a dual-limit strategy using staticLimit decorators to cap Tree Height (17) and Total Node Count (150), ensuring model stability without sacrificing complexity.

3. **JIT Compilation in Restricted Sandboxes**
DEAP's compile function runs in a restricted environment that initially prevented CuPy from accessing the system imports needed for Just-In-Time (JIT) GPU kernel compilation.

*Solution:* I applied the pset.context to include builtins, granting CuPy the necessary permissions to compile the evolved logic directly on the GPU hardware

## 📊 Performance & Results

The model was evolved over **70 generations** with a population of **1,000 individuals**.

**Training Dataset:** 95,804 samples.

**Best Fitness (Accuracy):** 73.97%

**Model Strategy:** *Elitism-based* evolutionary strategy(ea_with_elitism), preserving the top 2 individuals every generation to prevent regression.

**Peak Accuracy:** Successfully reached an accuracy of 73.97% on the training set.

**Final Output:** Classified 95,804 test samples into binary outputs (0 or 1) based on the best-evolved symbolic expression.

## 🛠️ Technical Stack
Language: Python 3.9

Evolutionary Logic: DEAP (Distributed Evolutionary Algorithms in Python)

GPU Acceleration: CuPy (optimized for CUDA)

Data Science: Pandas, NumPy

## 📂 Repository Contents
**evolution_analysis:** The primary research notebook containing the vectorized evaluation logic and evolutionary logs.

**gp_classifier_gpu.py:** Clean, modularized script for production runs.

**submission.csv:** Final classifications for the 95,804 test samples.


## 🧠 Optimization Insight: The "Sandbox" Fix
Genetic Programming in DEAP compiles trees using eval(), which runs in a restricted context. To enable CuPy's Just-In-Time (JIT) compilation of GPU kernels, the compilation context was modified to explicitly include builtins, allowing the GPU to dynamically compile the mathematical expressions evolved by the algorithm.