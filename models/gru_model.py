from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    return model
