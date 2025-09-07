import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('haberman.csv')
df.head()
df.columns = ['Age', 'Year', 'Nodes', 'Survival']
df.head()
df.shape
df.info()

df['survival'] = df['Survival'].map({1: 'yes', 2: 'no'})
print(df['survival'].value_counts())

#Detail about patients who survived
status_yes = df[df['survival'] == 'yes']
print(status_yes.describe())

#Visualize survived patients according to age
sns.histplot(data=df, x='Age', hue='survival', multiple='stack')
plt.legend()
plt.show()

# Calculate CDF for 'Nodes'
nodes = np.sort(df['Nodes'])
cdf = np.arange(1, len(nodes)+1) / len(nodes)

plt.figure(figsize=(8, 5))
plt.plot(nodes, cdf, marker='.', linestyle='none')
plt.xlabel('Number of Nodes')
plt.ylabel('CDF')
plt.title('CDF of Nodes')
plt.grid(True)
plt.show()

sns.jointplot(data=df, x='Year', y='Age', kind='kde', fill=True)
plt.title('Violin Plot of Age by Survival')
plt.xlabel('Survival')
plt.ylabel('Age')
plt.show()

sns.boxplot(data=df, x='survival', y='Nodes')
plt.title('Nodes Distribution by Survival')
plt.show()

sns.pairplot(df, hue='survival')
plt.suptitle('Pairplot of Features by Survival', y=1.02)
plt.show()

sns.heatmap(df[['Age', 'Year', 'Nodes']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

df['AgeGroup'] = pd.cut(df['Age'], bins=[29, 40, 50, 60, 70, 83], labels=['30-40', '41-50', '51-60', '61-70', '71-83'])
survival_rate = df.groupby('AgeGroup')['survival'].value_counts(normalize=True).unstack()
survival_rate.plot(kind='bar', stacked=True)
plt.title('Survival Rate by Age Group')
plt.ylabel('Proportion')
plt.show()

