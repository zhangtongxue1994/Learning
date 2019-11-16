import random
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import bisect


# 计算两个城市之间的距离
def calculate_distance(location1_xy, location2_xy):
    x_1, y_1 = location1_xy[0], location1_xy[1]
    x_2, y_2 = location2_xy[0], location2_xy[1]

    return math.sqrt(pow(x_1 - x_2, 2) + pow(y_1 - y_2, 2))


# 产生初始种群列表
def generate_initial(city_num, population_num):
    population_initial = []
    for i in range(population_num):
        child = list(range(city_num))
        random.shuffle(child)
        population_initial.append(child)

    return population_initial


# 计算城市距离矩阵
def distance_p2p_mat(location):
    distance_mat = []
    for i in range(location.shape[0]):
        dis_mat_each = []
        for j in range(20):
            dis = calculate_distance(location[i], location[j])
            dis_mat_each.append(dis)
        distance_mat.append(dis_mat_each)

    return distance_mat


# 计算每个个体的适应度值:路径长度的倒数
def calculate_adaptive(distance_mat, population):
    adaptive = []
    population_num = len(population)  # 种群数量
    city_num = len(distance_mat)
    for i in range(population_num):
        dis = 0
        for j in range(city_num - 1):
            dis = distance_mat[population[i][j]][population[i][j + 1]] + dis
        dis = distance_mat[population[i][19]][population[i][0]] + dis  # 回家
        dis_adp_each = 1.0 / dis
        adaptive.append(dis_adp_each)

    return adaptive


# 轮盘赌选择
def choose_fromlast(adaptive, population):
    adaptive = [pow(a, 1) for a in adaptive]  # 乘以15次方的原因是让好的个体被选取的概率更大
    adaptive_sum = np.sum(adaptive)  # 所有适应度之和
    population_num = len(population)  # 种群数量

    # 产生一个圆盘：概率累加
    prob = []
    prob_sum = 0
    for i in range(population_num):
        each_prob = adaptive[i] / adaptive_sum  # 每个个体所占的概率
        prob_sum += each_prob
        prob.append(prob_sum)  # 形成圆盘

    #  轮盘赌选择适应性强的个体
    population_choose = []
    for p in range(population_num):
        rand = random.uniform(0, 1)  # 产生随机数
        # 查找位置,二分查找http://www.voidcn.com/article/p-vbbuukpu-bsx.html
        child_index = bisect.bisect(prob, rand)
        population_choose.append(population[child_index])

    return population_choose


# 交叉算子
def cross_pronew(population, cross_prob):
    city_num = len(population[0])
    population_cross = []
    population_cross_num = int(len(population) * cross_prob)
    # 将个体两两一组划分，进行交叉
    for p in range(population_cross_num // 2):
        # 选出父本 母本
        child_1 = population[2 * p]
        child_2 = population[2 * p + 1]

        # 两点交叉的端点
        point_1 = random.randint(0, city_num - 2)
        point_2 = random.randint(point_1 + 1, city_num - 1)

        # 挑出来要交叉的片段（包括point_1，point_2）
        temp_1 = child_1[point_1:point_2 + 1]
        temp_2 = child_2[point_1:point_2 + 1]

        # 进行交叉
        child_1_new = []
        child_2_new = []
        for i in range(point_1):  # point_1（不包括）之前的部分
            if child_1[i] in temp_2:
                child_1_new.append(temp_1[temp_2.index(child_1[i])])
            else:
                child_1_new.append(child_1[i])
        child_1_new.extend(temp_2)
        for i in range(point_2 + 1, city_num):  # point_2（不包括）之后的部分
            if child_1[i] in temp_2:
                child_1_new.append(temp_1[temp_2.index(child_1[i])])
            else:
                child_1_new.append(child_1[i])

        for i in range(point_1):  # point_1（不包括）之前的部分
            if child_2[i] in temp_1:
                child_2_new.append(temp_2[temp_1.index(child_2[i])])
            else:
                child_1_new.append(child_1[i])
        child_2_new.extend(temp_1)
        for i in range(point_2 + 1, city_num):  # point_2（不包括）之后的部分
            if child_2[i] in temp_1:
                child_2_new.append(temp_2[temp_1.index(child_2[i])])
            else:
                child_2_new.append(child_2[i])

        population_cross.append(child_1)
        population_cross.append(child_2)

    return population_cross


# 变异算子
def var_pronew(population, variation_prob):
    population_variation = []
    num = len(population)
    for i in range(num):
        child = population[i]
        rand = random.uniform(0, 1)
        if rand < variation_prob:
            # 产生两个随机整数进行交换
            point_1 = random.randint(0, city_num - 1)
            point_2 = random.randint(0, city_num - 1)
            child[point_1], child[point_2] = child[point_2], child[point_1]

            population_variation.append(child)
        else:
            population_variation.append(child)

    return population_variation


if __name__ == "__main__":

    population_num = 300  # 随机生成的初始解的总数
    location = np.loadtxt('city_location.txt')
    city_num = location.shape[0]  # 城市总数

    iterations = 1000  # 遗传算法迭代次数
    cross_prob = 0.8  # 交叉概率
    variation_prob = 0.5  # 变异概率
    min_dis = []  # 记录历史最优距离
    best_child = []  # 记录历史最优个体

    # 产生初始种群
    population_initial = generate_initial(city_num=city_num, population_num=population_num)

    # 计算距离矩阵
    distance_mat = distance_p2p_mat(location=location)

    # 计算种群的适应度
    adaptive = calculate_adaptive(distance_mat=distance_mat, population=population_initial)

    # 记录历史最优距离和最优个体
    dis = 1.0 / min(adaptive)
    min_dis.append(dis)
    best_child = population_initial[adaptive.index(min(adaptive))]

    # 产生新一代个体
    population_new = copy.copy(population_initial)
    adaptive_new = copy.copy(adaptive)

    # 遗传算法开始
    for i in range(iterations):
        # 选择
        population_choose = choose_fromlast(adaptive=adaptive_new, population=population_new)
        # 交叉
        population_cross = cross_pronew(population=population_choose, cross_prob=cross_prob)
        # 变异
        population_variation = var_pronew(population=population_cross, variation_prob=variation_prob)
        # 产生新一代个体
        population_new = population_variation.copy()
        # 由于交叉概率影响，新一代个体数量减少，需要增加一部分上一代的较优个体，保持population_new的数量为population_num
        population_new.extend(population_choose[0:population_num - len(population_new)])
        adaptive_new = calculate_adaptive(distance_mat=distance_mat, population=population_new)
        # 更新历史最优距离和最优路径
        dis = 1.0 / min(adaptive_new)
        if dis < min_dis[-1]:
            min_dis.append(dis)
            best_child = population_variation[adaptive_new.index(min(adaptive_new))]
        else:
            min_dis.append(min_dis[-1])

    print('最短路线距离：', min_dis[-1])
    print('最优路线', best_child)

    plt.figure()
    # 绘制每代最短路径
    ax1 = plt.subplot(121)
    plt.plot(min_dis)
    plt.xlabel("iterations")
    plt.ylabel("distance")

    # 绘制路线
    ax2 = plt.subplot(122)
    plot_x_set = []
    plot_y_set = []
    for i in best_child:
        plot_x_set.append(location[i][0])
        plot_y_set.append(location[i][1])
    plt.plot(plot_x_set, plot_y_set, 'k')
    plt.scatter(plot_x_set, plot_y_set, )
    for i, txt in enumerate(best_child):
        plt.annotate(txt + 1, (plot_x_set[i], plot_y_set[i]))
    x = [plot_x_set[0], plot_x_set[-1]]
    y = [plot_y_set[0], plot_y_set[-1]]
    plt.plot(x, y, 'k')
    plt.title("Route")

    plt.show()
