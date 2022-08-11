import math
import operator
import matplotlib.pyplot as plt
import treePlotter

def createDataset():
    # 17个样本，6个属性
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]

    # 特征值列表（索引）
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']
    return dataSet,labels

#计算信息熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)#数据中中的样本个数
    labelCounts={}#统计每个类label出现的次数
    for featVec in dataSet:
        currentLable=featVec[-1]

        if currentLable not in labelCounts.keys():
            labelCounts[currentLable]=0#若数据集中没有，则加入新样本
        labelCounts[currentLable]+=1
    shannonEnt=0#信息熵
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*math.log(prob,2)

    return shannonEnt

#返回子集中所有取值为value的项，划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet=[]#保存筛选出的样本

    for featVec in dataSet:#特征向量，即数据集中的所有样本

        if featVec[axis]==value:#如果对应特征的取值和value相等，就保存到retdata中
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])#切片操作去除axis项
            retDataSet.append(reducedFeatVec)#加入列表

    return retDataSet

#挑选最优特征，即信息增益最大的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1#数据集目前特征的数量，-1是去除“好瓜”/“坏瓜”属性标签labels
    baseEntropy=calcShannonEnt(dataSet)#计算信息熵
    bestInfoGain=0.0#设初始增益为0
    bestFeature=-1

    for i in range(numFeatures):#遍历所有特征，i为属性的索引labels
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0

        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)#sub_dataset划分后的子集
            prob=len(subDataSet)/float(len(dataSet))#计算子集的概率
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy#增益=前后信息熵之差

        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i#如果当前增益大于之前最好，则进行更新
    return  bestFeature

#获取labels最多的那一类
def majorityCnt(classList):
    classCount={}#使用字典进行计数
    for vote in classList:
        if vote not in classCount.keys():#若当前值不在字典中，则创建一个key
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回一个列表，列表中的元组包含字典每一个item的键和值，依据这个元素出现次数的多少进行降序排序
    print(type(sortedClassCount))
    print(sortedClassCount)
    return sortedClassCount[0][0]#返回出现最多次的元素

#创建决策树
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]#取数据集中最后一个值为labels
    if classList.count(classList[0])==len(dataSet):
        return classList[0]
    # list[0]即“好瓜”label，若当前count等于数据集中好瓜数量（当前样本集内全是好瓜），则直接递归返回
    if len(dataSet[0])==1:#每个属性只能被使用一次，若只剩下一个属性，则一定是“好瓜”/“坏瓜”，返回此时训练集中数量多的类
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)#选择最好的分类属性的索引labels
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}#将分类结果以字典形式保存
    del(labels[bestFeat])#删除用过的属性（不能使用第二次）
    featValues=[example[bestFeat] for example in dataSet]#得到属性所有的取值
    uniqueVals=set(featValues)#使用集合操作进行去重
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree#对每个分支进行递归操作生成树
dataSet,labels=createDataset()
myTree=createTree(dataSet,labels)
treePlotter.createPlot(myTree)
print(myTree)

#使用测试特征测试决策树预测能力
def predict(Tree , test_data):
    first_feature = list(Tree.keys())[0]#从根部开始判断
    second_dict = Tree[first_feature]
    input_first = test_data.get(first_feature)
    input_value = second_dict[input_first]
    if isinstance(input_value , dict): #判断分支还是不是字典
        class_label = predict(input_value, test_data)#递归判断是否决策完成
    else:
        class_label = input_value
    return class_label

#验证
test_data_1 = {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '稍糊', '脐部': '凹陷', '触感': '硬滑'}
test_data_2 = {'色泽': '乌黑', '根蒂': '稍蜷', '敲声': '浊响', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑'}
result1 = predict(myTree,test_data_1)
result2 = predict(myTree,test_data_2)
print(result1)
print(result2)
