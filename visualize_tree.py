#Tout d'abord importer le jeu de données Iris 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris['data'][:, 2:]
y = iris['target']
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

export_graphviz( #en utilisant la méthode export_graphviz()qui fournit en sortie un fichier de définition graphique appelé iris_tree.dot
 tree_clf,
 out_file=open("iris_tree.dot","w"),
 feature_names=iris['feature_names'][2:],
 class_names=iris['target_names'],
 rounded=True,
 filled=True
)

#en executant maintenant notre script, nous obtenons un fichier dot nomme iris_tree

# convertissant le fichier dot en fichier image plus lisible 
# dot -Tpng iris_tree.dot -o iris_tree.png
