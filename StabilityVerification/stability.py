import sys
import os
curr_dir=os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.dirname(curr_dir)
sys.path.append(parent_dir)
from MyModel import Model1,Model2
class IfR1Positive:
    '''
    这一部分验证的是r1分别为正、负时，模型的预测能力
    结果应为：当r1为正时，模型的预测能力更强
    '''
    def __init__(self,r1,r7,r8):
        self.r1_positive=[x for x in r1 if x > 0]
        self.r7_positive=[r7[i] for i in range(len(r7)) if r1[i] > 0]
        self.r8_positive=[r8[i] for i in range(len(r8)) if r1[i] > 0]

        self.r1_negative=[x for x in r1 if x <= 0]
        self.r7_negative=[r7[i] for i in range(len(r7)) if r1[i] <= 0]
        self.r8_negative=[r8[i] for i in range(len(r8)) if r1[i] <= 0]

    def fit(self):
        model1_positive=Model1(self.r8_positive,self.r1_positive)
        model2_positive=Model2(self.r8_positive,self.r1_positive,self.r7_positive)
        model1_negative=Model1(self.r8_negative,self.r1_negative)
        model2_negative=Model2(self.r8_negative,self.r1_negative,self.r7_negative)
        model1_positive.fit()
        model1_negative.fit()
        model2_positive.fit()
        model2_negative.fit()
        self.model1_positive_R2=model1_positive.getR2()
        self.model1_negative_R2=model1_negative.getR2()
        self.model2_positive_R2=model2_positive.getR2()
        self.model2_negative_R2=model2_negative.getR2()
    def getResult(self):
        print('当r1为正时，两个模型的R^2分别为:',self.model1_positive_R2,self.model2_positive_R2)
        print('当r1为负时，两个模型的R^2分别为:',self.model1_negative_R2,self.model2_negative_R2)
        return self.model1_positive_R2,self.model2_positive_R2,self.model1_negative_R2,self.model2_negative_R2
    def Success(self):
        if self.model1_positive_R2>self.model1_negative_R2 and self.model2_positive_R2>self.model2_negative_R2:
            return True
        else:
            return False
