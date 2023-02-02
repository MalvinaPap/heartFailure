import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def cev(pca, n_comp):
    #cumulative explained variance plot
    plt.plot(pca.explained_variance_ratio_.cumsum(), lw=2, color='r')
    plt.title('Cumulative Expl. Variance by # of Principal Components \n', size=20)
    plt.xlabel('Number of Principal Components')
    plt.locator_params(axis='x', nbins=n_comp)
    plt.locator_params(axis='y', nbins=5)
    plt.ylabel('Variance Explained')
    plt.show()


def rankFeatures(df):
    X = df.drop('DEATH_EVENT', axis=1)
    feature_list = list(X.columns)
    y = df['DEATH_EVENT']
    n_comp=10
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=42)

    ss = StandardScaler()
    Xtrain = ss.fit_transform(Xtrain)
    Xtest = ss.transform(Xtest)

    pca = PCA(n_components=n_comp).fit(Xtrain)
    
    #explained variance ratio per principal component
    print('---------------------------------------------------')
    print('Explained Variance Ratio Per Principal Componet: ')
    print(pca.explained_variance_ratio_)
    print('---------------------------------------------------')
    #For every feature sum the eigenvalues from each principal component to find a final rank
    rankings = {}

    sums=pca.components_.sum(axis=0)

    for j in range(11):
        rankings[feature_list[j]]=sums[j]

    print('Ranking of Features based on sum of Eigenvalues: ')
    #sort features by importance
    for ranking in sorted(rankings, key=rankings.get, reverse=True):
        print(ranking, round(rankings[ranking],4))
    print('---------------------------------------------------')

    #call function to print plot
    cev(pca, n_comp)