import pandas as pd
import inspect
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def grid_search(model,para_grid, dataset):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(model, para_grid, cv=3, scoring='accuracy',n_jobs=10, return_train_score=True,verbose=1)
    grid_search.fit(dataset.drop('Transported', axis=1), dataset['Transported'])
    print('Best parameters: ', grid_search.best_params_)
    print('Best cross-validation score: ', grid_search.best_score_)
    return grid_search.best_params_
    
def make_para_grid():
    para_grid = {}
    
    sig = inspect.signature(RandomForestClassifier)
    params = sig.parameters
    defaults = {name: param.default for name, param in params.items() if param.default is not param.empty}

    print(defaults)
    
    para_grid.update({'n_estimators': list(range(50, 100, 2))})
    para_grid.update({'max_depth': list(range(1, 10))})
    para_grid.update({'min_samples_split': list(range(1, 10))})
    para_grid.update({'min_samples_leaf': list(range(1, 10))})
    # para_grid.update({'bootstrap': [True, False]})
    return para_grid

def fill_noneInt_data(data:pd.DataFrame, categorical_columns):
    for col in categorical_columns:
        if not col in data.keys():
            raise ValueError(f"Column {col} not found in dataset")
        unique_categories = data[col].unique()
        new_map={}
        for i, value in enumerate(unique_categories):
            new_map[value] = i
        data[col] = data[col].map(new_map)
        
    return data

if __name__ == "__main__":
    PRINT = False   
    PLOT = False
    current_file_path = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file_path)

    test_dataset_path= os.path.join(current_directory,"spaceship-titanic", "test.csv")
    train_dataset_path= os.path.join(current_directory,"spaceship-titanic", "train.csv")
    test_dataset= pd.read_csv(test_dataset_path)
    train_dataset= pd.read_csv(train_dataset_path)
    if PRINT:
        print("Test dataset: ", test_dataset.shape)
        print("Train dataset: ", train_dataset.shape)
        print(train_dataset.head())
        print(train_dataset.describe())
    if PLOT:
        plot_data = train_dataset.Destination.value_counts()
        plot_data.plot(kind='bar',title='Destination', color='b')
        fig, ax = plt.subplots(5,1,  figsize=(10, 10))
        # plt.subplots_adjust(top = 4)

        sns.histplot(train_dataset['Age'], color='b', bins=50, ax=ax[0]);
        sns.histplot(train_dataset['FoodCourt'], color='b', bins=50, ax=ax[1]);
        sns.histplot(train_dataset['ShoppingMall'], color='b', bins=50, ax=ax[2]);
        sns.histplot(train_dataset['Spa'], color='b', bins=50, ax=ax[3]);
        sns.histplot(train_dataset['VRDeck'], color='b', bins=50, ax=ax[4]);
        plt.show()
    
    # Data preprocessing
    train_dataset = train_dataset.drop(['PassengerId', 'Name'],axis=1)
    if PRINT:
        print(train_dataset.isnull().sum().sort_values(ascending=False))

    train_dataset[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = train_dataset[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
    if PRINT:
        print(train_dataset.isnull().sum().sort_values(ascending=False))
    train_dataset['Transported'] = train_dataset['Transported'].astype(int)
    train_dataset['VIP'] = train_dataset['VIP'].astype(int)
    train_dataset['CryoSleep'] = train_dataset['CryoSleep'].astype(int)
    
    
    train_dataset[["Deck", "Cabin_num", "Side"]] = train_dataset["Cabin"].str.split("/", expand=True)
    train_dataset = train_dataset.drop('Cabin', axis=1)
    print(train_dataset.head())
    #convert str to int
    train_dataset = fill_noneInt_data(train_dataset, ['Destination', 'HomePlanet','Deck', 'Side'])

    
    rf = RandomForestClassifier()
    grid = make_para_grid()
    best_params = grid_search(rf, grid, train_dataset)
    rf = rf.set_params(**best_params)
    rf.fit(train_dataset.drop('Transported', axis=1), train_dataset['Transported'])
    a,b = train_test_split(train_dataset, test_size=0.2, random_state=42)
    print(rf.score(b.drop('Transported', axis=1), b['Transported']))
    for k in zip(a.drop('Transported', axis=1).keys(), rf.feature_importances_):
        print(k)
    PassengerId = test_dataset.PassengerId
    test_dataset = test_dataset.drop(['PassengerId', 'Name'], axis=1)
    test_dataset[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = test_dataset[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
    test_dataset['VIP'] = test_dataset['VIP'].astype(int)
    test_dataset['CryoSleep'] = test_dataset['CryoSleep'].astype(int)
    test_dataset[["Deck", "Cabin_num", "Side"]] = test_dataset["Cabin"].str.split("/", expand=True)
    test_dataset = test_dataset.drop('Cabin', axis=1)
    test_dataset = fill_noneInt_data(test_dataset, ['Destination', 'HomePlanet','Deck', 'Side'])
    pred = rf.predict(test_dataset)
    pred = (pred > 0.5).astype(bool)
    output = pd.DataFrame({'PassengerId':PassengerId ,
                       'Transported': pred.squeeze()})
    output_path = os.path.join(current_directory,"spaceship-titanic", "submission.csv")
    output.to_csv(output_path, index=False)
    print(output.head())