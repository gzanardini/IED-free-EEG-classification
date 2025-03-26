import numpy as np 
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import precision_recall_curve

wandb.login(key='96e9a92e52e807ed253b3872afd1de1bafc3640a')

wandb.init(project='test', name=f'bbb_{np.random.randint(0, 100)}')

y_true = np.random.randint(0, 2, 50)
y_scores = np.random.rand(50)
precision, recall, _ = precision_recall_curve(y_true, y_scores)

plt.plot(recall, precision)
plt.xlabel('Recall')    
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
wandb.log({"pr_curve": plt})
plt.show()

