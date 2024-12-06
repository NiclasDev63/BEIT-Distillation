import matplotlib.pyplot as plt
largevalues = [13.16, 56.82, 58.74, 59.89, 61.41, 61.65, 62.21, 63.23, 64.12, 65.11, 64.83, 65.21, 65.79]
largeepochs = [0, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]

smallvalues = [0, 47.33, 49.48, 51.42, 52.56, 53.64, 54.21, 54.68, 54.87, 57.13, 58.53, 59.36, 60.4, 61, 61.29, 61.3, 61.68]
smallepochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

studentvalues = [0, 56.81, 59.35, 60.24, 60.66, 61.33, 61.65, 61.17, 61.75, 62.14, 62.15]
studentepochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

ax = plt.gca()
ax.set_ylim([0, 100])

plt.title("Student model performance Finetuning vs KD")

plt.xlabel("Training Epoch")
plt.ylabel("VizWiz Test Accuracy")

plt.plot(smallepochs, smallvalues, 'r')
plt.plot(studentepochs, studentvalues, 'b--')
plt.plot([0,16],[65.79, 65.79], 'k:')

ax.legend(["Finetuning", "Knowledge Distillation", "Teacher accuracy"])

plt.show()