import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
pca = PCA()  # just compute all of the components
X = pd.read_csv("/home/derek/workspace/deep_protein_binding/binding_test_hidden_features.csv").values
y = pd.read_csv("/home/derek/workspace/deep_protein_binding/binding_test_hidden_features_labels.csv").values


components = pca.fit_transform(std_scaler.fit_transform(X))
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

#TODO: insert a legend...

for i in range(10):
    with plt.style.context(('seaborn-muted')):
        plt.scatter(components[:, 0], components[:, i], c=y, s=1, cmap="Accent", marker="x")
        plt.xlabel("$PC_0$ (% explained variance {0:.2f})".format(explained_variance[0]))
        plt.ylabel("$PC_"+str(i)+"$ (% explained variance {0:.2f})".format(explained_variance[i]))
        plt.show()
    #plt.savefig("hidden_pca.svg")

