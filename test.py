datasets = np.loadtxt('wine.txt',delimiter=',')
labels,datas= datasets[:,0],datasets[:,1:]
changelabels(labels) #对标签进行处理，处理后的标签为0,1,2等等
datas = (datas - datas.min(axis=0))/(datas.max(axis=0) - datas.min(axis=0)) #各个属性进行归一化处理


labelnum = np.shape(list(set(labels)))[0]
attrinum = np.shape(datas)[1]
samplenum = np.shape(labels)[0]
featurenum = labelnum*attrinum

#参数初始化
weights = np.zeros(featurenum)

#特征的先验期望
priorfeatureE = getpriorfeatureE(labels,datas)

#用L-BFGS优化算法求解最优参数
weights,likelyfuncvalue = lbfgs(fun = funcobject,gfun = getgradients,x0 = weights,datas = datas,labels = labels,priorfeatureE = priorfeatureE,m=10,maxk = 20)

#根据训练的参数预测各个样本的类别
predictlabels = getpredictlabels(datas,weights)
