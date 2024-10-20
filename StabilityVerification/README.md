## 稳健性分析 —— 条件预测能力
  根据研报分析，第一个半小时收益率(即r1)的正负性会影响模型的预测能力（即模型的R^2）
## 代码功能
  在stability.py中，定义了一个叫做IfR1Positive的类，初始化参数为r1,r7,r8，分别代表第一个半小时，倒数第二个半小时，最后一个半小时的收益率；fit()功能用于使用上述数据来进行回归，getResult()功能用于获取r1分别为正负时模型的R^2，最后的Success()和Fail()分别用于判断当前股票是否符合研报的预测。
  




## 结果分析
对于两个模型，我们分别使用460支股票进行回归，并得到了回归的R^2分布，结果如下
!["model 1"](https://github.com/shirz22/FBDQAMidTermWork/blob/main/StabilityVerification/model1.png)
!["model 2"](https://github.com/shirz22/FBDQAMidTermWork/blob/main/StabilityVerification/model2.png)
#### 我们可以明显看到，当r1为正时，模型的R^2整体向右偏，这意味着当r1为正时，整体来讲模型的预测能力会较高
#### 但是结果显示，完全符合预测（也就是当r1为正时，两个模型的R^2均高于r1为负时的两个模型的R^2）的股票只占据了24%，而反之完全不符合的股票占据了46%。这在一定程度上说明了美股和A股存在的一定程度的差异
