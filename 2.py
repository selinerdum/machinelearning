import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Örnek tahminler
y_true = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1]
y_pred = [0, 0, 0, 1, 1, 0, 1, 0, 1, 1]

cnf_matrix = confusion_matrix(y_true, y_pred)

# Sınıf isimleri
class_names = [0, 1]

# Heatmap oluşturma
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')

# Eksen etiketlerini ve başlığı ayarlama
plt.xticks(np.arange(len(class_names)), class_names)
plt.yticks(np.arange(len(class_names)), class_names)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title('Confusion matrix', y=1.1)

plt.show()
