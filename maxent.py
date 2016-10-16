def changelabels(labels):
    #对标签进行处理，使得第一个标签为0，第二个标签为1，依次递增
    index = 0
    labeldict = dict()
    for i in range(np.shape(labels)[0]):
        if labels[i] not in labeldict:#如果该标签未曾出现过
            labeldict[labels[i]] = index#在字典中加入该标签
            index += 1
        labels[i] = labeldict[labels[i]]   

def getprobs(data,weights):
    #计算当前参数下data属于各个类别的概率
    attrinum = np.shape(data)[0]
    labelnum = np.shape(weights)[0]/attrinum

    exps = np.zeros(labelnum)
    for label in range(labelnum):
        exps[label] = np.exp(np.dot(weights[int(label*attrinum):int((label+1)*attrinum)],data))
    exps /= np.sum(exps)
    return exps


def getpriorfeatureE(labels,datas):
    # 计算特征函数的先验分布的期望值
    labelnum = np.shape(list(set(labels)))[0]
    attrinum = np.shape(datas)[1]
    samplenum = np.shape(labels)[0]
    featurenum = labelnum*attrinum

    priorfeatureE = np.zeros(featurenum)
    for label in range(labelnum):
        #print datas[labels==label]
        priorfeatureE[int(label*attrinum):int((label+1)*attrinum)] = np.sum(datas[labels==label],axis=0)
    priorfeatureE /= 1.0*samplenum
    return priorfeatureE

def getpostfeatureE(datas,weights):
    #计算特征函数的后验分布的期望值
    attrinum = np.shape(datas[0])[0]
    labelnum = np.shape(weights)[0]/attrinum
    samplenum = np.shape(datas)[0]

    postfeatureE = np.zeros(featurenum)
    for i in range(samplenum):
        probs = getprobs(datas[i],weights)
        for label in range(labelnum):
            postfeatureE[int(label*attrinum):int((label+1)*attrinum)] += 1.0/samplenum*probs[label]*datas[i]
    return postfeatureE

def getgradients(datas,weights,priorfeatureE):
    #计算当前参数下的各参数的梯度向量

    postfeatureE = getpostfeatureE(datas,weights)

    return postfeatureE-priorfeatureE

def funcobject(labels,datas,weights):
    # 目标函数是对对数似然函数取负，故要使其最小化
    labelnum = np.shape(list(set(labels)))[0]
    samplenum = np.shape(labels)[0]

    part0 = 0
    part1 = 0
    for i in range(samplenum):
        part0 += 1.0/samplenum*np.log(np.sum([np.exp(np.dot(datas[i],weights[int(label*attrinum):int((label+1)*attrinum)])) for label in range(labelnum)]))
        part1 += 1.0/samplenum*np.dot(datas[i],weights[int(labels[i]*attrinum):int((labels[i]+1)*attrinum)])

    return part0 - part1  

def getpredictlabels(datas,weights):
    # 根据训练得到的参数预测样本的类别
    samplenum = np.shape(datas)[0]
    predictlabels = np.array([np.nan]*samplenum)
    for i in range(samplenum):
        probs = getprobs(datas[i],weights)
        label = probs.argmax()
        predictlabels[i] = label
    return predictlabels 

def twoloop(s, y, rho,gk):
    # 被lbfgs函数调用
    n = len(s) #向量序列的长度

    if np.shape(s)[0] >= 1:
        #h0是标量，而非矩阵
        h0 = 1.0*np.dot(s[-1],y[-1])/np.dot(y[-1],y[-1])
    else:
        h0 = 1

    a = np.empty((n,))

    q = gk.copy() 
    for i in range(n - 1, -1, -1): 
        a[i] = rho[i] * np.dot(s[i], q)
        q -= a[i] * y[i]
    z = h0*q

    for i in range(n):
        b = rho[i] * np.dot(y[i], z)
        z += s[i] * (a[i] - b)

    return z   

def lbfgs(fun,gfun,x0,datas,labels,priorfeatureE,m=5,maxk = 20):
    # fun和gfun分别是目标函数及其一阶导数,x0是初值,m为储存的序列的大小
    rou = 0.55
    sigma = 0.4
    epsilon = 1e-5
    k = 0
    n = np.shape(x0)[0] #自变量的维度

    s, y, rho = [], [], []

    while k < maxk :
        gk = gfun(datas,x0,priorfeatureE)
        if np.linalg.norm(gk) < epsilon:
            break

        dk = -1.0*twoloop(s, y, rho,gk)

        m0=0;
        mk=0
        while m0 < 20: # 用Armijo搜索求步长
            if fun(labels,datas,x0+rou**m0*dk) < fun(labels,datas,x0)+sigma*rou**m0*np.dot(gk,dk): 
                mk = m0
                break
            m0 += 1


        x = x0 + rou**mk*dk
        sk = x - x0
        yk = gfun(datas,x,priorfeatureE) - gk   

        if np.dot(sk,yk) > 0: #增加新的向量
            rho.append(1.0/np.dot(sk,yk))
            s.append(sk)
            y.append(yk)
        if np.shape(rho)[0] > m: #弃掉最旧向量
            rho.pop(0)
            s.pop(0)
            y.pop(0)

        k += 1
        x0 = x

    return x0,fun(labels,datas,x0)#,k#分别是最优点坐标，最优值，迭代次数
    
    
 
