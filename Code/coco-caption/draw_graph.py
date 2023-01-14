import matplotlib.pyplot as plt
import numpy as np

name_list = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
data_list = [[],[],[],[],[],[],[]]

# x = ['1', '2', '3' ,'4' ,'5' ,'6', '7', '8', '9', '10']
x = [1, 2, 3, 5, 7, 9]

for i in x:

    path = f"./data_res/N_trainable_mean_0.016-coco_prefix-00{i - 1}.txt"
    rfile = open(path, 'r')

    for j, line in enumerate(rfile):
        cur = line.split()
        data_list[j].append(float(cur[1]))

    rfile.close()

print(data_list)


plt.figure()
plt.title('Trainable Mean')
plt.xlabel('Epoch')  
plt.ylabel('Value')  
for i in range(7):
    plt.plot(x, data_list[i], marker='o', markersize=3)
for i in range(7):
    for a, b in zip(x, data_list[i]):
        if i == 1 or i == 4:
            plt.text(a, b, b, ha='center', va='top', fontsize=8)
        else:
            plt.text(a, b, b, ha='center', va='bottom', fontsize=8)
        # break
plt.legend(name_list)
plt.show()
plt.savefig('./graphs/11.jpg')
