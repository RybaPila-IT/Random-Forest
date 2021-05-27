from sklearn import preprocessing


def dataframe_encode(data):

    encoders = [preprocessing.LabelEncoder().fit(data[column]) for column in data]

    for column, encoder in zip(data, encoders):
        data[column] = encoder.transform(data[column])

    return encoders


def series_encode(data):

    encoder = preprocessing.LabelEncoder().fit(data)
    data = encoder.transform(data)

    return encoder


def dataframe_decode(data, decoders):

    for column, decoder in zip(data, decoders):
        data[column] = decoder.inverse_transform(data[column])


def series_decode(data, decoder):
    return decoder.inverse_transform(data)
