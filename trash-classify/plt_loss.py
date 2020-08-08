# Print accurancy curve and loss curve
import csv
import matplotlib.pyplot as plt 
from matplotlib.pyplot import MultipleLocator


loss = []
val_loss = []
acc = []
val_acc = []

with open('result_show/M_b_Xception_896.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        loss.append(float(row['loss']))
        val_loss.append(float(row['val_loss']))
        acc.append(float(row['acc']))
        val_acc.append(float(row['val_acc']))



epochs = range(1, len(loss) + 1)


plt.ylim([0,5])
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.plot(epochs, loss, 'lightseagreen', label='Training-loss', marker='.', linestyle='-')
plt.plot(epochs, val_loss, 'indianred', label='val-loss', marker='.', linestyle='-')
plt.title('Train-loss and Val-loss ', fontsize=20)
plt.legend(fontsize=20)
plt.grid()


plt.figure()
plt.ylim([0,1])
y_major_locator=MultipleLocator(0.1)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.plot(epochs, acc, 'lightseagreen', label='Training-acc', marker='.', linestyle='-')
plt.plot(epochs, val_acc, 'indianred', label='val-acc', marker='.', linestyle='-')
plt.title('Train-acc and Val-acc', fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.show()
