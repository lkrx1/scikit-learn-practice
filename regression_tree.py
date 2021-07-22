from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz,DecisionTreeRegressor

iris = load_iris()
X = iris['data'][:, 2:]
y = iris['target']
#appliquons la classe DecisionTreeRegressor a notre base iris
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X,y)

#la méthode export_graphviz() qui fournit en sortie un fichier de définition graphique appelé iris_tree_regressor.dot
export_graphviz(
 tree_reg,
 out_file=open("iris_tree_regressor.dot","w"),
 feature_names=iris['feature_names'][2:],
 class_names=iris['target_names'],
 rounded=True,
 filled=True
)

# conversion de notre fichier dot en format png lisible
# dot -Tpng iris_tree_regressor.dot -o iris_tree_regressor.png