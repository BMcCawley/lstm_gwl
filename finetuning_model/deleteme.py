"""
I am just trying to picture the input data into an LSTM trained on multiple catchments
The first dimension is the batch size. Here, we have a batch size of 2.
The second dimension is the sequence length. Here, we have a sequence length of 3.
The third dimension is the number of features. Here, we have 2 features.
"""
input_data = (
    [
        [
            [0, 0.1],
            [0.5, 9],
            [3, 5],
        ],
        [
            [2, 8],
            [0, 9],
            [1, 0.2],
        ],
    ],
)
