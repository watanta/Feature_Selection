
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.data import wine_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

X, y = wine_data()
X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                   stratify=y,
                                                   test_size=0.3,
                                                   random_state=1)

knn = KNeighborsClassifier(n_neighbors=2)

sfs1 = SFS(estimator=knn,
           k_features=(3, 10),
           forward=True,
           floating=False,
           scoring='accuracy',
           cv=5)

pipe = make_pipeline(StandardScaler(), sfs1)

pipe.fit(X_train, y_train)

print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
print('all subsets:\n', sfs1.subsets_)
plot_sfs(sfs1.get_metric_dict(), kind='std_err');


import pandas as pd
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
import lightgbm as gbm


X = pd.DataFrame(np.random.rand(10, 10))
y = pd.DataFrame(np.random.rand(10))

my_model = gbm.LGBMRegressor(random_state=0)

perm = PermutationImportance(my_model, random_state=1, cv=5).fit(X.values, y.values.flatten())

perm_feat_imp = perm.feature_importances_


from feature_selection.wrapping_method.permutation_score import get_permutation_score
from feature_selection.wrapping_method.permutation_score import perm_selection

import pandas as pd
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
import lightgbm as gbm


X = pd.DataFrame(np.random.rand(10, 10))
y = pd.DataFrame(np.random.rand(10))

my_model = gbm.LGBMRegressor(random_state=0)

perm = perm_selection(X, y, my_model, 5, cv=5, random_state=1)

import shap
import numpy as np
import pandas as pd
import lightgbm as gbm
from sklearn.model_selection import KFold
from feature_selection.embedded_method.shap_value import get_shap_value_moment
from feature_selection.embedded_method.shap_value import shap_value_moment_selection_rank


X = pd.DataFrame(np.random.rand(10, 10))
y = pd.DataFrame(np.random.rand(10))

my_model = gbm.LGBMRegressor(random_state=0)

shap_value = get_shap_value_moment(X, y, my_model)

shap_df = shap_value_moment_selection_rank(X, y, my_model, 5)

import numpy as np
import pandas as pd
import lightgbm as gbm
from feature_selection.feature_make.arithmetic import plus

X = pd.DataFrame(np.random.rand(10, 10))
y = pd.DataFrame(np.random.rand(10))

df = plus(X)

