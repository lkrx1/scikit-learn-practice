from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris['data'][:, 2:]
y = iris['target']
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

#probabilite de prediction d'une fleur dont les pétales mesurent 5 cm de long et 1,5 cm de large
print(tree_clf.predict_proba([[5, 1.5]]))

# prédiction de la classe, il devrait indiquer Iris versicolor (classe 1) puisque c’est celle qui a la plus forte probabilité
# print(tree_clf.predict([[5, 1.5]]))