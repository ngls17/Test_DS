import numpy as np
import pandas as pd


class PreparerTSV:

    def __init__(self, train, test, type_):
        if type_ == 'url':
            tmp1 = pd.read_csv('https://drive.google.com/uc?export=download&id=' + train.split('/')[-2],
                               sep='\t', chunksize=1000000)
            tmp2 = pd.read_csv('https://drive.google.com/uc?export=download&id=' + test.split('/')[-2],
                               sep='\t', chunksize=1000000)
            self.train_df = pd.concat(tmp1, ignore_index=True)
            self.test_df = pd.concat(tmp2, ignore_index=True)
        elif type_ == 'file':
            tmp1 = pd.read_csv(train, sep='\t', chunksize=1000000)
            tmp2 = pd.read_csv(test, sep='\t', chunksize=1000000)

            self.train_df = pd.concat(tmp1, ignore_index=True)
            self.test_df = pd.concat(tmp2, ignore_index=True)

        self.train_col_len = len(self.train_df.columns)
        self.test_col_len = len(self.test_df.columns)
        self.test_df_norm = None

    def separate_features(self):
        """ Разбивка features на тип признака и признаки"""
        self.train_df = self.separate_df(self.train_df)
        self.train_col_len = len(self.train_df.columns)

        self.test_df = self.separate_df(self.test_df)
        self.test_col_len = len(self.test_df.columns)

        if self.train_col_len == self.test_col_len:
            self.test_df_norm = pd.DataFrame(columns=range(self.train_col_len))
        else:
            raise ValueError('Train and Test has different dimentions')

    def stand(self, type_):
        """ Стандартизация исходных признаков"""
        if type_ == 'z_score':
            mean = np.mean(self.train_df.iloc[:, 2:].to_numpy(dtype='int16', copy=False), axis=0)
            std = np.std(self.train_df.iloc[:, 2:].to_numpy(dtype='int16', copy=False), axis=0)

            self.test_df_norm = ((self.test_df.iloc[:, 2:].astype('int16', copy=False) - mean) / std)
            self.test_df_norm['id_job'] = self.test_df['id_job']

            self.test_df_norm.rename(lambda x: f'feature_{self.get_feature_type()}_stand_{x}' if x in range(
                len(self.test_df_norm.columns)) else x, axis=1, inplace=True)
            cols = ['id_job'] + [col for col in self.test_df_norm if col != 'id_job']
            self.test_df_norm = self.test_df_norm[cols]

        else:
            raise TypeError('Stand type is not supported')

    @staticmethod
    def separate_df(dataframe):
        """
        Этот метод разбивает features на тип признака и признаки
        :param dataframe:
            pandas.DataFrame
            Columns:
                Name: id_job, dtype: object
                Name: feature, dtype: object

        :return: dataframe:
            pandas.DataFrame
            Columns:
                Name: id_job, dtype: object
                Name: feature_type, dtype: object
                Name: 0,  dtype: object,
                ...
                Name: 255,  dtype: object,
        """
        dataframe = pd.concat([dataframe['id_job'], dataframe['features'].str.split(',', expand=True)],
                              axis=1)
        dataframe_col_len = len(dataframe.columns)
        dataframe.rename({0: 'feature_type'}, axis=1, inplace=True)
        dataframe.rename(lambda x: x - 1 if x in range(dataframe_col_len) else x, axis=1, inplace=True)
        return dataframe

    def get_feature_type(self):
        """ Проверка всех вакансий на один и тот же идентификатор набора признаков"""
        if len(self.train_df['feature_type'].unique()) == len(self.test_df['feature_type'].unique()) == 1:
            return self.train_df['feature_type'].unique()[0]
        else:
            raise TypeError('Feature type different for jobs')

    def get_max_feature_index(self):
        """ Поиск индекса i максимального значения признака feature_2_{i} для вакансий"""
        feature_type = self.get_feature_type()
        self.test_df_norm[f'max_feature_{feature_type}_index'] = np.argmax(
            self.test_df.iloc[:, 2:].to_numpy(dtype='int16', copy=False), axis=1)

    def get_max_feature_2_abs_mean_diff(self):
        """ Вычисление абсолютного отклонения признака с индексом max_feature_2_index от его среднего значения"""
        feature_type = self.get_feature_type()
        max_value = np.max(self.test_df.iloc[:, 2:].to_numpy(dtype='int16', copy=False), axis=1)
        self.test_df_norm[f'max_feature_{feature_type}_abs_mean_diff'] = max_value - np.mean(
            self.test_df.iloc[:, 2:].to_numpy(dtype='int16', copy=False), axis=1)
