from md_input import dense_to_one_hot, load_csv_without_header
import numpy as np
from tensorflow.python.framework import dtypes
import csv

input_sample = '/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/raw_data_no_duplication_now/inputsam.csv'

r = csv.reader(open(input_sample))  # Here your csv file
row_count = sum(1 for row in r)

test_set = load_csv_without_header(
    filename=input_sample,
    target_dtype=np.int,
    features_dtype=np.int,
    n_samples=row_count,
    block_size=2,
)

print(test_set)