# 1 最大最小聚类算法

算法流程：

![图片](https://user-images.githubusercontent.com/34503835/123507589-e4650500-d69c-11eb-953a-e242f86e837a.png)

参考链接：https://blog.csdn.net/guyuealian/article/details/53708042

# 2.1 k-means聚类算法

参考链接：https://www.cnblogs.com/lc1217/p/6893924.html

算法流程：
  
  1）随机选取k个点作为种子点(这k个点不一定属于数据集)
  
  2）分别计算每个数据点到k个种子点的距离，离哪个种子点最近，就属于哪类
 
  3）重新计算k个种子点的坐标(简单常用的方法是求坐标值的平均值作为新的坐标值)
  
  4）重复2、3步，直到种子点坐标不变或者循环次数完成

不足：
  
  1）初始分类数目k值很难估计，不确定应该分成多少类才最合适(ISODATA算法通过类的自动合并和分裂，得到较为合理的类型数目k。这里不讲这个算法)
  
  2）不同的随机种子会得到完全不同的结果(K-Means++算法可以用来解决这个问题，其可以有效地选择初始点)
  
# 2.2 k-means++聚类算法

参考链接：https://www.cnblogs.com/lc1217/p/6893924.html

算法流程：
 
 1）在数据集中随机挑选1个点作为种子点
 
 2）计算剩数据点到这个点的距离d(x),并且加入到列表

 3）再取一个随机值。这次的选择思路是：先取一个能落在上步计算的距离列表求和后(sum(dis_list))的随机值rom，然后用rom -= d(x)，直到rom<=0，此时的点就是下一个“种子点”

 4）重复第2步和第3步，直到选出k个种子
 
 5）进行标准的K-Means算法

# 2.3 python自带的k-means聚类算法

参考链接：https://zhuanlan.zhihu.com/p/47916491

# 3 层级聚类算法

算法流程：
凝聚层次聚类与k均值的关键不同。 我们不是选择多个聚类并以随机质心开始，而是从数据集中的每个点开始作为“聚类”。然后我们找到两个最接近的点并将它们组合成一个聚类。 然后，我们找到下一个最近的点，这些点成为一个簇。 我们重复这个过程，直到我们只有一个巨大的巨型集群。

参考链接：https://zhuanlan.zhihu.com/p/47916491
