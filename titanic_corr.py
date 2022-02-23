import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df = pd.read_csv('data/titanic.csv')
print(titanic_df.info())

fig, ax = plt.subplots(1,2)
titanic_df['survived'][titanic_df['sex'] == 'male'].value_counts().plot.\
    pie(ax=ax[0], autopct='%1.1f%%')
titanic_df['survived'][titanic_df['sex'] == 'female'].value_counts().plot.\
    pie(ax=ax[1], autopct='%1.1f%%')
ax[0].set_title('Male')
ax[1].set_title('FeMale')
plt.show()

sns.countplot('pclass', data=titanic_df, hue='survived')
plt.show()

titanic_corr = titanic_df.corr(method='pearson')
print(titanic_corr)

titanic_corr.to_csv('data/titanic_corr.csv', index=False)