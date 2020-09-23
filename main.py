from time import time

from preparer.preparer import PreparerTSV


def print_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        print('Procedure takes: ', time() - start)

    return wrapper


@print_time
def main():
    train_url = 'https://drive.google.com/file/d/16NjEaAaGIguNaB1UyZ3L4qXB36zoGcaw/view?usp=sharing'
    test_url = 'https://drive.google.com/file/d/1qg2zGNlFGN3lWjehi2byAGiW_jm-aL6h/view?usp=sharing'
    source_type = 'url'

    df = PreparerTSV(train_url, test_url, source_type)
    df.separate_features()
    df.stand(type_='z_score')
    df.get_max_feature_index()
    df.get_max_feature_2_abs_mean_diff()
    df.test_df_norm.to_csv('test_proc.tsv', sep='\t')


if __name__ == '__main__':
    main()
