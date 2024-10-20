# 对比模型
# 文件清单
- `CompareModels`
    - `compareModel.ipynb`:对于1000支股票的2021-2023三年的全部交易日的数据，采用多种回归模型进行对比分析
    - `README.md`:说明以及分析文档
    - `image.png`:结果图片
# 步骤以及结果
## 1.多种回归对比分析
### 文件
`compareModel.ipynb`
### 内容
对于1000支股票的2021-2023三年的全部交易日的数据，采用多种回归模型进行对比分析，包括OLS，岭回归，Lasso回归，基函数回归，KNN，SVM，随机森林
### 结果
详情见`compareModel.ipyn`，在此展示对比分析图

![alt text](image.png)
### 分析
结果来看，无论是一个还是两个参数，只有线性模型的R2大于0，非线性拟合的结果均显著小于0，其中SVM小于0的程度最大。这说明r8和r7，r1的关系非常弱，只有非常细微的线性关系，基本类似于随机噪点的形式，非线性模型极易过拟合，回归的效果很差。
![image](https://github.com/user-attachments/assets/587f0fff-731c-40cb-bcf3-45fb03481e10)
![image](https://github.com/user-attachments/assets/d3266807-6ff8-49d3-802a-a708ac6cbf13)
