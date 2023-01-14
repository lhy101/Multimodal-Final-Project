import matplotlib.pyplot as plt
import numpy as np


name_list = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
data_list = [[],[],[],[],[],[],[]]

# x = ['1', '2', '3' ,'4' ,'5' ,'6', '7', '8', '9', '10']
x = ['Text Only', '4:1', '2:1', 'Image Only']

path = f"./data_res1/N_0_0.016_with_norm-coco_prefix-009.txt"
rfile = open(path, 'r')

for j, line in enumerate(rfile):
    cur = line.split()
    data_list[j].append(float(cur[1]))

rfile.close()

path = f"./data_res/U_0_sqrt_0.001-coco_prefix-001.txt"
rfile = open(path, 'r')

for j, line in enumerate(rfile):
    cur = line.split()
    data_list[j].append(float(cur[1]))

rfile.close()

path = f"./data_res/U_0_sqrt_0.016-coco_prefix-009.txt"
rfile = open(path, 'r')

for j, line in enumerate(rfile):
    cur = line.split()
    data_list[j].append(float(cur[1]))

rfile.close()

path = f"./data_res/U_0_sqrt_0.1-coco_prefix-002.txt"
rfile = open(path, 'r')

for j, line in enumerate(rfile):
    cur = line.split()
    data_list[j].append(float(cur[1]))

rfile.close()



print(data_list)

x = np.arange(4)  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
for i in range(-3, 4):
    rects = ax.bar(x + i * width, data_list[i + 3], width, label=name_list[i + 3])
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Effect of Using Uniform Noise Instead of Gaussian Noise')
ax.set_xticks(x)
x = ['Baseline', '0.001', '0.016', '0.1']
ax.set_xticklabels(x)
ax.legend()
plt.xlabel('Variance')
plt.ylabel('Value')
plt.show()

plt.savefig('./graphs/12.jpg')


'''
plt.figure()
plt.title('Average Loss on Each Epoch when Training ')
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
x = ['1', '2', '3' ,'4' ,'5' ,'6', '7', '8', '9', '10']
data_list = [1.9605594747031538, 1.4395594747428335, 1.3036620362794051, 1.2142651434403087, 1.1471600078926225, 1.0935200059478907, 1.0483006578845715, 1.010197544929608, 0.9784042260581318, 0.9534021509447154]
plt.plot(x, data_list, label='Ours')
data_list = [1.7560547043461765, 1.202023373785509, 1.058531985834551, 0.9663717258311613, 0.8978883351358222, 0.8430039371913955, 0.7988447298393262, 0.7621024429536246, 0.7321953948437899, 0.7083126375222243]
plt.plot(x, data_list, label='Baseline')
plt.legend()
plt.savefig('./graphs/8.jpg')
'''