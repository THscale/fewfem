#https://qiita.com/Sunset_Yuhi/items/4c4ddc25609a7619cce0
#を参考にした
#1次元Poisson方程式を、有限要素法で解く
#d/dx[p(x)du(x)/dx]=f(x) (x_min<x<x_max)
#u(x_min)=alpha, du(x_max)/dx=beta
#を改良する

#
# delta_u/delta_t + a delta_u/ delta_x - k delta^2_u/delta_x^2 = f
#を解く
#

import time  #時刻を扱うライブラリ
import numpy as np  #NumPyライブラリ
import scipy.linalg  #SciPyの線形計算ライブラリ
import matplotlib.pyplot as plt  #データ可視化ライブラリ
import csv
import math
import datetime
import os
import makemodel


#境界条件u(x_min)=alpha, du(x_max)/dx=beta。
#alphaやbetaが"inf"の時は、何も処理しないようにする。

boundary_condition_list1 = [0, 1.0, "inf"]
boundary_condition_list2 = [50, 0.0, "inf"]

#傾き条件の場合
#boundary_condition_list2 = [node_total_num-1, "inf", 1.0]


#計算モードの設定　将来的にはファイルにしたい

#
# delta_u/delta_t + a delta_u/ delta_x - k delta^2_u/delta_x^2 = f
#を解く

#ファイル、計算の初期条件のまとめ

#nodeの座標は下記ファイルに書き込む

constant_a = 1
constant_k = 0.01
constant_f = 0
node_ele_file_name = "node.csv"
initial_u_file_name =  "initial.csv"
delta_t = 0.01
total_t_step = 100
time1_t_step = 10
time2_t_step = 20
time3_t_step = 50

#このDensityとは、各step、nodeにおいて間の数を示す。整数である必要がある。
#time_step_density:その値だけ倍する
#node_density:各nodeの間のnodeの個数+1の値
time_step_density = 10;
node_density = 10;


#境界条件をdensityに関して返還

boundary_condition_list1[0] = boundary_condition_list1[0]*(node_density-1)+boundary_condition_list1[0]
boundary_condition_list2[0] = boundary_condition_list2[0]*(node_density-1)+boundary_condition_list2[0]

#結果を出すファイル名をつくる
now = datetime.datetime.now()
result_file_name = "result/result_file_" + now.strftime('%Y%m%d_%H%M%S') + '.csv'

#ここでファイルをつくる
f = open(result_file_name, mode='w')
f.close()

#ここでデータを抜く
#これはCSVファイルをIncludeする関数
def include_csv(file_name):
    include_list = []
    with open(file_name) as csvfile:
        for row in csv.reader(csvfile):
            include_list.append(row)
    return include_list


#ここで時間のtime stepに関する変換を行う
#時間の変換,delta_tのみ異なることに注意
time1_t_step = makemodel.time_step_density_adjustment(time1_t_step,time_step_density)
time2_t_step = makemodel.time_step_density_adjustment(time2_t_step,time_step_density)
time3_t_step = makemodel.time_step_density_adjustment(time3_t_step,time_step_density)
total_t_step = makemodel.time_step_density_adjustment(total_t_step,time_step_density)
delta_t = makemodel.time_delta_density_adjustment(delta_t,time_step_density)

#


#各node、座標を読み込む
node_ele_file = include_csv(node_ele_file_name)
node_ele_coordinates = node_ele_file[1]
node_ele_coordinates = [float(s) for s in node_ele_coordinates]
#ここでnumpy形式に変換する
node_ele_coordinates = np.array(node_ele_coordinates,np.float64)

#densityに関する返還をする
node_ele_coordinates = makemodel.coordinate_density_adjustment(node_ele_coordinates,node_density)

#print("Node (node_ele_coordinates)")
#print(node_ele_coordinates)
#各ノードの座標を順番に書く

#ノード、エレメントの個数
node_total_num = len(node_ele_coordinates)
ele_total_num = node_total_num - 1


#各エレメント左右のノード番号を示している。

def make_node_number_of_each_element(node_number_file):
    node_number_file = np.empty((ele_total_num,2), np.int) #各要素のGlobal節点番号
    for e in range(ele_total_num):
        node_number_file[e,0] = e
        node_number_file[e,1] = e+1
    return node_number_file

print("Node number of each elements")
node_num_glo_in_seg_ele = np.empty((ele_total_num,2), np.int)
node_num_glo_in_seg_ele = make_node_number_of_each_element(node_num_glo_in_seg_ele)
#ここで、各エレメントの左右のノード番号を割り振っている（座標ではないことに注意）

#print(node_num_glo_in_seg_ele)

######初期のuを読み込む


initial_u_file = include_csv(initial_u_file_name)
vec_initial_u = initial_u_file[1]
vec_initial_u = [float(s) for s in vec_initial_u]
#密度に応じて変換
vec_initial_u = makemodel.coordinate_density_adjustment(vec_initial_u,node_density)


print("initial u")
print(vec_initial_u)

vec_u_n = np.zeros((ele_total_num,2), np.float64)
vec_u_next = np.zeros((ele_total_num,2), np.float64)
vec_u_reslut = np.zeros((total_t_step+1,node_total_num), np.float64)


vec_u_n = vec_initial_u
vec_u_reslut[0] = vec_initial_u

########## 要素行列を求める ##########
#各線分要素の長さを計算

ele_total_num = len(node_ele_coordinates)-1
element_length = np.empty(ele_total_num, np.float64)

#print(ele_total_num)

for e in range(ele_total_num):
   element_length[e] = np.absolute(node_ele_coordinates[e+1] - node_ele_coordinates[e])

#print("Element length")
#print(element_length)
#各Elementの長さを示す

############result fileの調整#########################

result_row = []
result_row = np.append("","node_ele_coordinates")
with open(result_file_name, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(result_row)

result_row = []
result_row = np.append(node_ele_coordinates, "Time")
with open(result_file_name, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(result_row)

result_row = []
result_row = np.append(vec_initial_u, 0)
with open(result_file_name, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(result_row)


#各要素行列の初期化

#要素行列の初期化
#要素係数行列(ゼロで初期化)
mat_M_ele = np.zeros((ele_total_num,2,2), np.float64)
mat_M_delta_ele = np.zeros((ele_total_num,2,2), np.float64)
vec_Q_ele = np.zeros((ele_total_num,2), np.float64)
vec_Q_delta_ele = np.zeros((ele_total_num,2), np.float64)
mat_S_ele = np.zeros((ele_total_num,2,2), np.float64)
mat_S_delta_ele = np.zeros((ele_total_num,2,2), np.float64)
mat_K_ele = np.zeros((ele_total_num,2,2), np.float64)
scalar_tau_ele = np.zeros((ele_total_num), np.float64)


#要素行列の各成分を計算
#print("Local matrix")

#将来的にtauの設定はモードに応じて色々変えられるようにしたい
for e in range(ele_total_num):
    scalar_tau_ele[e] = ((2/delta_t)**2+(2*constant_a/element_length[e])**2)**-0.5

#for e in range(ele_total_num):
#    scalar_tau_ele[e] = ((2*constant_a/element_length[e])**2)**-0.5


for e in range(ele_total_num):
            mat_M_ele[e,0,0] = element_length[e]/3
            mat_M_ele[e,0,1] = element_length[e]/6
            mat_M_ele[e,1,0] = element_length[e]/6
            mat_M_ele[e,1,1] = element_length[e]/3


for e in range(ele_total_num):
    mat_M_delta_ele[e, 0, 0] =  -1 * scalar_tau_ele[e] * constant_a * 1/2
    mat_M_delta_ele[e, 0, 1] =  -1 * scalar_tau_ele[e] * constant_a * 1/2
    mat_M_delta_ele[e, 1, 0] =  1 * scalar_tau_ele[e] * constant_a * 1/2
    mat_M_delta_ele[e, 1, 1] =  1 * scalar_tau_ele[e] * constant_a * 1/2

#    for i in range(2):
#        for j in range(2):
#            mat_M_delta_ele[e,i,j] = 1 * scalar_tau_ele[e] * constant_a * ((-1)**j)/2

#実は計算力学2版には誤植がある
#mat_M_delta_ele[e,0,0] = - tau a (1/2)
#mat_M_delta_ele[e,0,1] = - tau a (1/2)
#mat_M_delta_ele[e,1,0] =   tau a (1/2)
#mat_M_delta_ele[e,1,1] =   tau a (1/2)
#

for e in range(ele_total_num):
    for i in range(2):
            vec_Q_ele[e,i] = constant_f * element_length[e] / 4

#なぜかここを正しい式である「/2」ではなく/4とすると正しい値となる。おそらく重ね合わせの問題だと思われるが詳細不明。
#しかも教科書の計算を精査すると最終的に下記になる気がするが、それはそれで±0になる
# for e in range(ele_total_num):
#        vec_Q_ele[e, 0] = constant_f * element_length[e] / 2
#        vec_Q_ele[e,1] = -1 * constant_f * element_length[e] / 2


#-1かけているのは、range2が1,2ではなく0,1だから
for e in range(ele_total_num):
    for i in range(2):
            vec_Q_delta_ele[e,i] = -1 * scalar_tau_ele[e] * constant_a *constant_f * ((-1)**i) /2

#ここ行列を重ね合わせると結局±0になるように見えるが……あとで検証する

#-1かけているのは、range2が1,2ではなく0,1だから
for e in range(ele_total_num):
    for i in range(2):
        for j in range(2):
            mat_S_ele[e, i, j] = -1 * constant_a * ((-1) ** j) / 2

for e in range(ele_total_num):
    for i in range(2):
        for j in range(2):
            mat_S_delta_ele[e,i,j] =  scalar_tau_ele[e] * (constant_a**2) * ((-1)**i)*((-1)**j)/element_length[e]


#mat_S_delta_ele[e,0,0] =   tau a^2 (1/h)
#mat_S_delta_ele[e,0,1] =  - tau a^2 (1/h)
#mat_S_delta_ele[e,1,0] =  - tau a^2 (1/h)
#mat_S_delta_ele[e,1,1] =    tau a^2 (1/h)
#

for e in range(ele_total_num):
    for i in range(2):
        for j in range(2):
            mat_K_ele[e,i,j] = (constant_k) * ((-1)**i)*((-1)**j)/element_length[e]




#print("scalar_tau_ele")
#print(scalar_tau_ele)
#print("mat_M_ele")
#print(mat_M_ele)
#print("mat_M_delta_ele")
#print(mat_M_delta_ele)
#print("vec_Q_ele")
#print(vec_Q_ele)
#print("vec_Q_delta_ele")
#print(vec_Q_delta_ele)
#print("mat_S_ele")
#print(mat_S_ele)
#print("mat_S_delta_ele")
#print(mat_S_delta_ele)
#print("mat_K_ele")
#print(mat_K_ele)

########## 全体行列を組み立てる ##########
#
#差分法のシータをどうするかにより複数あるが、本質的に
#
#mat_A_glo * u_n+1 = mat_B_glo * u_n + vec_b_glo
#
#
#全体行列の初期化
mat_A_glo = np.zeros((node_total_num,node_total_num), np.float64) #全体係数行列(ゼロで初期化)
mat_B_glo = np.zeros((node_total_num,node_total_num), np.float64)

#全体係数ベクトル(ゼロで初期化)
vec_b_glo = np.zeros(node_total_num, np.float64)
vec_right_side = np.zeros((node_total_num), np.float64)

#要素行列から全体行列を組み立てる

#要素行列から全体行列を組み立てる
#ここからＣｒａｎｋ－　Ｎｉｃｏｌｓｏｎ、前進法、後進法を分岐する

#ここからCrank-Nicolson

for e in range(ele_total_num):
    for i in range(2):
        for j in range(2):
            mat_A_glo[ node_num_glo_in_seg_ele[e,i], node_num_glo_in_seg_ele[e,j] ] += (mat_M_ele[e,i,j] + mat_M_delta_ele[e,i,j])/delta_t + (mat_S_ele[e,i,j]+mat_S_delta_ele[e,i,j]+mat_K_ele[e,i,j])/2
            mat_B_glo[ node_num_glo_in_seg_ele[e,i], node_num_glo_in_seg_ele[e,j] ] += (mat_M_ele[e,i,j] + mat_M_delta_ele[e,i,j])/delta_t - (mat_S_ele[e,i,j]+mat_S_delta_ele[e,i,j]+mat_K_ele[e,i,j])/2
            vec_b_glo[ node_num_glo_in_seg_ele[e,i] ] += vec_Q_ele[e,i]+vec_Q_delta_ele[e,i]

#ここでmat_A_gloは、node数×node数であることに注意（element数ではない）

#境界条件の計算では、境界条件の補正をする「前」のmat_A_gloの値を使用するために、ここでinitialとして保存する。
#なお、=でコピーすると両方参照になるので注意
mat_A_glo_initial = mat_A_glo.copy()

#ここから時間ごとの計算を連続で行う。
#
#

current_step_time = 0
for t in range(total_t_step):
    ####右辺（つまり mat_B_glo * u_n + vec_b_gloを計算する）
    vec_right_side = mat_B_glo @ vec_u_n + vec_b_glo
#    vec_right_side = np.dot(mat_B_glo , vec_u_n ) + vec_b_glo
#と同じ計算をしている
    #行列どおしの掛け算は@らしい

    #print("Global計算の後")
    #print(mat_A_glo)
    #print(mat_B_glo)
#    print("vec_b_glo")
#    print(vec_b_glo)
    #print(vec_right_side)

    #任意の節点に境界条件を実装する
    def boundary(node_num_glo, Dirichlet, Neumann):
        #ディリクレ境界条件
        if (Dirichlet!="inf"):  #Dirichlet=無限大の時は処理しない
            #print("matA")
            #print(mat_A_glo)
            #print("fitst")
            #print(vec_right_side)
            #print("mat_A_glo[node_num_glo,:]")
            #print(mat_A_glo[node_num_glo,:])
            vec_right_side[:] = vec_right_side[:]-Dirichlet*mat_A_glo_initial[node_num_glo,:]  #定数ベクトルに行の値を移項
            #print("second")
            #print(vec_right_side)
            vec_right_side[node_num_glo] = Dirichlet  #関数を任意の値で固定
            #print("third")
            #print(vec_right_side)
            #print("-----")
            mat_A_glo[node_num_glo,:] = 0.0  #行を全て0にする
            mat_A_glo[:,node_num_glo] = 0.0  #列を全て0にする
            mat_A_glo[node_num_glo,node_num_glo] = 1.0  #対角成分は1にする

         #ノイマン境界条件
#       if (Neumann!="inf"):  #Neumann=無限大の時は処理しない
#            vec_right_side[node_num_glo] += Neumann #関数を任意の傾きで固定

    print("mat_A_initial1")
    print(mat_A_glo_initial)
    print("mat_A_glo1")
    print(mat_A_glo)


    boundary(boundary_condition_list1[0],boundary_condition_list1[1],boundary_condition_list1[2])
    boundary(boundary_condition_list2[0],boundary_condition_list2[1],boundary_condition_list2[2])

    print("mat_A_initial2")
    print(mat_A_glo_initial)
    print("mat_A_glo2")
    print(mat_A_glo)

    #境界条件による補正

    #print(mat_A_glo)
    #print(vec_b_glo)
    #print(vec_right_side)

############### 連立方程式を解く ###############

    vec_u_next = scipy.linalg.solve(mat_A_glo,vec_right_side)  #Au=bから、未知数ベクトルuを求める
    vec_u_n = vec_u_next
    vec_u_reslut[t+1] = vec_u_n
    #ここで、0行目には、初期を入れているため、+1している

    current_step_time = current_step_time + delta_t

    result_row = []
    result_row = np.append(vec_u_reslut[t+1], current_step_time)
    with open(result_file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_row)


############### 計算結果を表示する ###############

#print(vec_u_reslut)

############### 計算結果を可視化 ###############
#plt.rcParams['font.family'] = 'Times New Roman'  #全体のフォントを設定
#plt.rcParams['font.size'] = 10  #フォントサイズを設定
#plt.rcParams['lines.linewidth'] = 2  # 線の太さ設定
#plt.title("Finite element analysis of 1D Poisson's equation")  #グラフタイトル

plt.xlabel("$x$")  #x軸の名前
plt.ylabel("$u(x)$")  #y軸の名前
plt.grid()  #点線の目盛りを表示


plt.plot(node_ele_coordinates,vec_initial_u, label="$\hat{u}(x)$ initial", color='#000080')  #折線グラフを作成
plt.plot(node_ele_coordinates,vec_u_reslut[time1_t_step], label="$\hat{u}(x)$ time1_t_step", color='#00bfff')  #折線グラフを作成
plt.plot(node_ele_coordinates,vec_u_reslut[time2_t_step], label="$\hat{u}(x)$ time2_t_step", color='#ff00ff')  #折線グラフを作成
plt.plot(node_ele_coordinates,vec_u_reslut[time3_t_step], label="$\hat{u}(x)$ time3_t_step", color='#ff0000')  #折線グラフを作成
plt.plot(node_ele_coordinates,vec_u_next, label="$\hat{u}(x)$ last", color='#000000')  #折線グラフを作成
#plt.scatter(node_ele_coordinates,vec_u_next)  #点グラフを作成
#plt.scatter(node_ele_coordinates,vec_u_reslut[time1_t_step])
#plt.scatter(node_ele_coordinates,vec_u_reslut[time2_t_step])
#plt.scatter(node_ele_coordinates,vec_u_reslut[time3_t_step])
#plt.scatter(node_ele_coordinates,vec_initial_u)

#近似解をプロット
#approximate_time1 = list(node_ele_coordinates)
#approximate_time2 = list(node_ele_coordinates)
#approximate_time3 = list(node_ele_coordinates)

#for i in range(node_total_num):
#    approximate_time1[i] = math.sin(math.pi * node_ele_coordinates[i]) * math.e **(-1* constant_k * math.pi ** 2 * time1_t_step *delta_t)
#    approximate_time2[i] = math.sin(math.pi * node_ele_coordinates[i]) * math.e **(-1* constant_k * math.pi ** 2 * time2_t_step *delta_t)
#    approximate_time3[i] = math.sin(math.pi * node_ele_coordinates[i]) * math.e **(-1* constant_k * math.pi ** 2 * time3_t_step *delta_t)

#    approximate_time1[i] = math.sin(math.pi * node_ele_coordinates[i]) * math.e **(-1* constant_k * math.pi ** 2 * time1_t_step *delta_t)

#     approximate_time1[i] = node_ele_coordinates[i] *2

#sa3 = list (approximate_time3-vec_u_reslut[time3_t_step])
#print(sa3)


#plt.scatter(node_ele_coordinates, approximate_time1)
#plt.scatter(node_ele_coordinates, approximate_time2)
#plt.scatter(node_ele_coordinates, approximate_time3)
#plt.scatter(node_ele_coordinates,vec_u_reslut[time2_t_step])
#plt.scatter(node_ele_coordinates,vec_u_reslut[time3_t_step])


#更に体裁を整える
plt.axis('tight') #余白を狭くする
plt.axhline(0, color='#000000')  #x軸(y=0)の線
plt.axvline(0, color='#000000')  #y軸(x=0)の線
plt.legend(loc='best')  #凡例(グラフラベル)を表示
#for n in range(node_total_num):  #節点番号をグラフ内に表示
#    plt.text(node_ele_coordinates[n],vec_u_next[n], n, ha='center',va='bottom', color='#0000ff')

plt.show()  #グラフを表示
#plt.savefig('fem1d_poisson.pdf')
#plt.savefig('fem1d_poisson.png')
