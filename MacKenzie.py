from matplotlib import pyplot as plt
import math
import random

#계측 데이터 그리기
logx = [73,297,-53,127,-224,-15,2,27,0,-61,67,-223,220,-91,115,60,59,62,53,-104,-773,-249,359,-186,-23,309,-454,56,40,-93,340,55,-337,119,275,-260,-199,356,-291,77,-138,-43,-6,-116,3,-251,260,-233,-82,480,-268,174,-217,-68,196,17,-28,-64,160,31,309,-277,207,20,-429,207,-76,41,-32,45,238,-432,9,304,39,140,-396,335,59,-63,-990,66,33,-92,142,-200,283,-181,63,229,29,-194,162,-347,78,285,0,-48,-86,66]
logy = [190,-98,302,97,-311,242,-86,-232,217,143,-267,-126,273,-114,-187,-4,329,-275,201,-285,-673,37,5,35,-113,79,-323,67,394,-144,142,-194,115,-91,-203,126,214,-236,-166,240,-252,115,18,-14,-269,168,75,-306,79,158,-78,161,-417,321,-294,-46,201,-80,-33,-71,-648,-164,99,-185,158,41,-40,153,87,-187,157,-388,5,301,-36,140,-128,-243,103,269,263,-82,-233,-64,380,-1,-427,-8,207,46,-71,31,-196,406,-342,25,376,-265,-219,62]
logm = [80,118,119,122,239,66,226,160,77,94,125,218,142,207,154,77,153,91,145,301,106,159,229,240,191,66,225,130,136,151,206,61,222,151,62,235,177,75,214,400,164,140,53,148,145,201,123,248,74,158,235,164,204,105,237,67,103,129,69,262,95,216,235,215,52,241,212,75,232,244,198,73,74,92,229,62,90,124,134,239,134,74,222,184,239,192,82,61,65,75,226,156,97,69,234,103,57,236,75]
logt = [809,975,700,703,644,739,565,666,659,634,678,633,649,562,646,602,683,672,638,682,738,618,568,495,482,614,706,569,634,565,585,663,550,504,1035,475,649,775,546,534,651,531,559,537,670,623,571,602,901,886,481,580,632,703,581,922,613,537,670,577,805,568,559,522,908,524,476,584,534,472,558,765,768,758,513,624,941,1014,622,677,818,614,558,551,580,536,743,756,724,992,528,621,677,856,540,715,967,537,786,573]
loga = []
train = []
def lognonaka(x, y, m):
    return (math.sqrt(x**2 + y**2) / m) + 1



for i in range(8): loga.append([0, 0])

for i in range(99):
    logid = lognonaka(logx[i], logy[i], logm[i])
    train.append([logid, logt[i]])
    plt.scatter(logid, logt[i], s = 1, c = 'k')
    if logid < 8 :
        logidavr = math.floor(logid)
        loga[logidavr][0] = loga[logidavr][0] + logt[i]
        loga[logidavr][1] = loga[logidavr][1] + 1

for i in range(8):
    if loga[i][1] > 0: plt.scatter(i, loga[i][0] / loga[i][1], s = 5, c = 'b')
    if i == 5: plt.text(i, loga[i][0] / loga[i][1], 'average')


# Linear Regression 부분
w = 0
b = 0

def mse(dataset, reg1, reg2):
    msev = 0
    for i in range(len(dataset)):
        msev = msev + (dataset[i][1] - (reg1 * math.log(dataset[i][0], 2)) - reg2) ** 2
    return msev / len(dataset)

def msedw(dataset, reg1, reg2):
    msedv = 0
    for i in range(len(dataset)):
        msedv = msedv + 2 * (reg1 * (math.log(dataset[i][0], 2) ** 2) - (dataset[i][1] * math.log(dataset[i][0], 2)) + reg2 * math.log(dataset[i][0], 2))
    return msedv / len(dataset)

def msedb(dataset, reg1, reg2):
    msedv = 0
    for i in range(len(dataset)):
        msedv = msedv + 2 * (reg2 - dataset[i][1] + (reg1 * math.log(dataset[i][0], 2)))
    return msedv / len(dataset)

for i in range(10000):
    w = w - (msedw(train, w, b) * 0.01)
    b = b - (msedb(train, w, b) * 0.01)

for i in range(1, 10):
    plt.scatter(i, (w * math.log(i, 2)) + b, s = 10, c = 'r')
    if i == 9:
        plt.text(i, (w * math.log(i, 2)) + b, "MSE: "+str(abs(mse(train, w, b))))

print("w:", w)
print("b:", b)
print(abs(mse(train, w, b)))

plt.title('DATA 5, 1600DPI', fontsize=25)
plt.xlabel('D/W+1')
plt.ylabel('milliseconds')

plt.show()