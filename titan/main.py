import pandas as pd
import inspect
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from train import start_train
from model import MyDataset, NeuralNetwork
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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
    
    para_grid.update({'n_estimators': list(range(70, 120, 2))})
    para_grid.update({'max_depth': list(range(1, 15))})
    para_grid.update({'min_samples_split': list(range(1, 10))})
    # para_grid.update({'min_samples_leaf': list(range(1, 10))})
    return para_grid

def fill_noneInt_data(data:pd.DataFrame, categorical_columns):
    for col in categorical_columns:
        if not col in data.keys():
            raise ValueError(f"Column {col} not found in dataset")
        
        unique_categories = data[col].unique()
        unique_categories= sorted(unique_categories, key=lambda x: str(x))
        print("----",unique_categories)
        new_map={}
        for i, value in enumerate(unique_categories):
            new_map[value] = i
        data[col] = data[col].map(new_map)
        
    return data

def fill_nan_base_on_correlation(train_test:pd.DataFrame):

    train_test.loc[:,['Room']] = train_test.PassengerId.apply(lambda x: x[0:4])

    guide_vip = train_test.loc[:,['Room','VIP']].dropna().drop_duplicates('Room')

    guide_cabin = train_test.loc[:,['Room','Cabin']].dropna().drop_duplicates('Room')

    guide_homePlanet = train_test.loc[:,['Room','HomePlanet']].dropna().drop_duplicates('Room')

    guide_destination = train_test.loc[:,['Room','Destination']].dropna().drop_duplicates('Room')

    #adding these to dataset
    train_test = pd.merge(train_test,guide_vip, how="left",on="Room",suffixes=('','_y'))
    train_test = pd.merge(train_test,guide_cabin, how="left",on="Room",suffixes=('','_y'))
    train_test = pd.merge(train_test,guide_homePlanet, how="left",on="Room",suffixes=('','_y'))
    train_test = pd.merge(train_test,guide_destination, how="left",on="Room",suffixes=('','_y'))
    train_test.loc[:,['VIP']] = train_test.apply(lambda x: x.VIP_y if pd.isna(x.VIP) else x, axis =1)
    train_test.loc[:,['Cabin']] = train_test.apply(lambda x: x.Cabin_y if pd.isna(x.Cabin) else x, axis =1)
    train_test.loc[:,['HomePlanet']] = train_test.apply(lambda x: x.HomePlanet_y if pd.isna(x.HomePlanet) else x, axis =1)
    train_test.loc[:,['Destination']] = train_test.apply(lambda x: x.Destination_y if pd.isna(x.Destination) else x, axis =1)
    
    train_test = train_test.drop(['VIP_y','Cabin_y','HomePlanet_y','Destination_y'], axis=1)
    return train_test


def preprocess_dataframe(dataFrame:pd.DataFrame,test_set = False):

    dataFrame = fill_nan_base_on_correlation(dataFrame)
     # Data preprocessing
    dataFrame[["Deck", "Cabin_num", "Side"]] = dataFrame["Cabin"].str.split("/", expand=True)
    dataFrame = dataFrame.drop('Cabin', axis=1)

    #feature engineering
    #Calculate total money spent by summing up individual expenditures 
    Expenses_columns = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    for column in Expenses_columns:
        dataFrame[column] = np.where((pd.isnull(dataFrame[column])) & (dataFrame['CryoSleep'] == True), 0.0, dataFrame[column])   
        
    dataFrame["MoneySpent"] = dataFrame[Expenses_columns].sum(axis=1)
    dataFrame.loc[:,['CryoSleep']] = dataFrame.apply(lambda x: True if x.MoneySpent == 0 and pd.isnull(x.CryoSleep) else x, axis =1)
    # dataFrame.loc[:,["VIP"]] = dataFrame.apply(lambda x: False if x['MoneySpent'] <= mean_value and pd.isnull(x.VIP) else True, axis=1)


    mean_value = dataFrame["MoneySpent"].mean()    
    # dataFrame.loc[dataFrame['MoneySpent'] <= mean_value, pd.isnull(dataFrame[columns_special_fill])] = 0  
    # dataFrame.loc[dataFrame['MoneySpent'] > mean_value, pd.isnull(dataFrame[columns_special_fill])] = 1   
    dataFrame.loc[(dataFrame['MoneySpent'] <= mean_value) & pd.isnull(dataFrame["VIP"]), "VIP"] = False
    dataFrame.loc[(dataFrame['MoneySpent'] > mean_value) & pd.isnull(dataFrame["VIP"]), "VIP"] = True
    dataFrame['VIP'] = dataFrame['VIP'].astype(int)

    
    
    num_cols = ['ShoppingMall','FoodCourt','RoomService','Spa','VRDeck','MoneySpent']
    cat_cols = ['CryoSleep','Deck','Side','HomePlanet','Destination']
    #notvg = ['HomePlanet','VIP','ShoppingMall','FoodCourt','Age','Cabin_2','Destination']
    num_imp = SimpleImputer(strategy='mean')
    cat_imp = SimpleImputer(strategy='most_frequent')
    ohe = OneHotEncoder(handle_unknown='ignore')


    dataFrame[num_cols] = pd.DataFrame(num_imp.fit_transform(dataFrame[num_cols]),columns=num_cols)
    dataFrame[cat_cols] = pd.DataFrame(cat_imp.fit_transform(dataFrame[cat_cols]),columns=cat_cols)
    
    # temp_train = pd.DataFrame(ohe.fit_transform(dataFrame[cat_cols]).toarray(), columns= ohe.get_feature_names_out())
    # dataFrame = dataFrame.drop(cat_cols,axis=1)
    # dataFrame = pd.concat([dataFrame,temp_train],axis=1)
    
    
    
    columns_mean_fill = ["Age"]
    dataFrame[columns_mean_fill] = dataFrame[columns_mean_fill].fillna(dataFrame[columns_mean_fill].mean())
    dataFrame['Cabin_num'] = dataFrame['Cabin_num'].fillna(0)
    #convert str to int
    dataFrame = fill_noneInt_data(dataFrame,cat_cols)
    print(dataFrame.head(15))
    
    if not test_set:
        dataFrame = dataFrame.drop(['PassengerId'],axis=1)
        dataFrame['Transported'] = dataFrame['Transported'].astype(int)
    dataFrame = dataFrame.drop(['Name'], axis=1)

    return dataFrame
def extra_data_preprocessing(dataFrame):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = StandardScaler()
    float_columns = dataFrame.select_dtypes(include=['float64','float32']).columns
    dataFrame[float_columns] = scaler.fit_transform(dataFrame[float_columns])
    return dataFrame

def UseRandomForest(train_dataset, test_dataset,user_grid_search = False):
    rf = RandomForestClassifier(n_estimators=115)
    if user_grid_search:
        grid = make_para_grid()
        best_params = grid_search(rf, grid, train_dataset)
        rf = rf.set_params(**best_params)
    rf.fit(train_dataset.drop('Transported', axis=1), train_dataset['Transported'])
    for k in zip(train_dataset.drop('Transported', axis=1).keys(), rf.feature_importances_):
        print(k)
    test_dataset = test_dataset.drop(['PassengerId'], axis=1)
    pred = rf.predict(test_dataset)

    return pred

def UsescikitNN(train_dataset, test_dataset):
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier()
    nn.fit(train_dataset.drop('Transported', axis=1), train_dataset['Transported'])
    a,b = train_test_split(train_dataset, test_size=0.2, random_state=42)
    print(nn.score(b.drop('Transported', axis=1), b['Transported']))
    test_dataset = test_dataset.drop(['PassengerId'], axis=1)
    pred = nn.predict(test_dataset)
    return pred


def UsetorchNN(train_dateset,test_dataset):
    train_labels =  train_dateset['Transported']
    train = train_dateset.drop('Transported', axis=1)
    train = extra_data_preprocessing(train)
    x_train, x_validation, y_train, y_validation = train_test_split(train, train_labels, test_size=0.2, shuffle=True, random_state=5)
    

    batch_size = 64
    train_dataloader = DataLoader(MyDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(MyDataset(x_validation, y_validation), batch_size=batch_size, shuffle=True)
    
    network= NeuralNetwork(input_size=x_train.shape[1]).to(torch.device('cuda'))
    output_net = start_train(network,train_dataloader,val_dataloader,torch.device('cuda'))
    test_dataset = test_dataset.drop(['PassengerId'], axis=1)
    test_dataset = extra_data_preprocessing(test_dataset)
    test_dataset = torch.from_numpy(np.array(test_dataset, dtype=np.float32)).to(torch.device('cuda'))
    output_net.eval()
    pred = output_net(test_dataset)
    return pred

    

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
    PassengerId = test_dataset.PassengerId
    train_dataset = preprocess_dataframe(train_dataset)
    test_dataset = preprocess_dataframe(test_dataset, test_set=True)

    pred_rf = UseRandomForest(train_dataset, test_dataset, user_grid_search=False)
    input("Press Enter to continue...")
    pred_sciNN = UsescikitNN(train_dataset, test_dataset)
    pred_torchNN =UsetorchNN(train_dataset, test_dataset).cpu().detach().numpy().squeeze()
   
    weight = [0.6, 0.1, 0.3]
    prd_list = [pred_rf, pred_sciNN, pred_torchNN]
    pred =weight[0]*prd_list[0] + weight[1]*prd_list[1] + weight[2]*prd_list[2]
    pred = (pred > 0.5).astype(bool)

    output = pd.DataFrame({'PassengerId':PassengerId,
                    'Transported': pred.squeeze()})
    
    output_path = os.path.join(current_directory,"spaceship-titanic", "submission.csv")
    output.to_csv(output_path, index=False)
    print(output.head())