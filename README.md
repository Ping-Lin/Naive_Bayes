#README

##machine learning project
* bayesian classifier with Laplace estimate
* 10-bin discretization(equal width)  
* using five-fold for training and testing
* SNB for feature selection
* Dirichlet and general Dirichlet to prove accuracy

## Usage
`python project_naive_bayes.py data data_index method random_seed````Ex: python project_naive_bayes.py dataset/hepatitis/hepatitis.data dataset/hepatitis/hepatitis_input_index.txt 2 101
```### Argument
1. data: original data
2. data_index(content) 
   * want attribute index
   * need discretization attribute index
   * class index
   * class type's number
3. method
   
   ```   method=1: No feature selection     method=2: feature selection  
   ``` 
4. random_seed: any number
​	