import sys
import keras
import keras.layers as layers
from keras.engine.topology import Layer

def output_feedback(n_a, n_dense, n_y):
    encoder_LSTM = layers.Bidirectional(layers.LSTM(units = n_a, return_state=True))
    decoder_LSTM = layers.Bidirectional(layers.LSTM(units = n_a, return_state=True))
    flatter = layers.Flatten()
    dense = layers.Dense(units = n_dense, activation='tanh')
    dense_out = layers.Dense(units = 1, activation='linear')
    concatenator = layers.Concatenate()
    concatenator_out = layers.Concatenate()
    reshapor = layers.Reshape((1, n_a*2))
    

    x_pdf_input = layers.Input(shape=(None, 1))
    x_x_input = layers.Input(shape=(None, 1))
    
    encoder_input = concatenator([x_pdf_input, x_x_input])
    
    _, for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c = encoder_LSTM(encoder_input)  
    
    decoder_input = layers.Input(shape=(1,n_a*2))
    deco_input = decoder_input
    decoder_state = [for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c]
    
    out_sequence = list()
    progress_steps = 20
    for i in range(n_y):
        sys.stdout.write("\rFeedback: {0} out of {1}".format(str(i), str(n_y)))
        sys.stdout.flush()
        
        decoder_output, for_deco_h, for_deco_c, back_deco_h, back_deco_c = decoder_LSTM(deco_input, initial_state=decoder_state)
        decoder_output = reshapor(decoder_output)
        decoder_output_flat = flatter(decoder_output)
        out = dense(decoder_output_flat)
        out = dense_out(out)
        
        out_sequence.append(out)
        
        deco_input = decoder_output
        decoder_state = [for_deco_h, for_deco_c, back_deco_h, back_deco_c]
        
    print()
    out_sequence = concatenator_out(out_sequence)
    model = keras.models.Model(inputs=[x_pdf_input, x_x_input, decoder_input], outputs=out_sequence)
    return model



def state_cell_only(n_a, n_dense, drop_rate, n_y):
    encoder_LSTM = layers.Bidirectional(layers.LSTM(units = n_a, return_state=True))
    decoder_LSTM_L1 = layers.Bidirectional(layers.LSTM(units = n_a, return_sequences=True))
    decoder_LSTM_L2 = layers.Bidirectional(layers.LSTM(units = n_a, return_sequences=True))
    flatter = layers.Flatten()
    dense_L1 = layers.Dense(units = n_dense, activation='tanh')
    dense_L2 = layers.Dense(units = n_dense, activation='tanh')
    dropout = layers.Dropout(rate=drop_rate)
    dense_out = layers.Dense(units = n_y, activation='linear')
    concatenator = layers.Concatenate()

    
    x_pdf_input = layers.Input(shape=(None, 1))
    x_x_input = layers.Input(shape=(None, 1))
    
    encoder_input = concatenator([x_pdf_input, x_x_input])
    
    _, for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c = encoder_LSTM(encoder_input)  
    
    decoder_input = layers.Input(shape=(1,n_a))
    decoder_state = [for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c]

    decoder_output = decoder_LSTM_L1(decoder_input, initial_state=decoder_state)
    decoder_output = decoder_LSTM_L2(decoder_output, initial_state=decoder_state)
    decoder_output_flat = flatter(decoder_output)
    out = dense_L1(decoder_output_flat)
    out = dropout(out)
    out = dense_L2(out)
    out = dropout(out)
    out = dense_out(out)
    
    model = keras.models.Model(inputs=[x_pdf_input, x_x_input, decoder_input], outputs=out)
    return model

def skipper(n_a, n_dense, drop_rate, n_y):
    encoder_LSTM = layers.Bidirectional(layers.LSTM(units = n_a, return_state=True))
    dense_encoder = layers.Dense(units = n_y)
    decoder_LSTM_L1 = layers.Bidirectional(layers.LSTM(units = n_a, return_sequences=True))
    decoder_LSTM_L2 = layers.Bidirectional(layers.LSTM(units = n_a, return_sequences=True))
    flatter = layers.Flatten()
    dense_L1 = layers.Dense(units = n_dense, activation='tanh')
    dense_L2 = layers.Dense(units = n_dense, activation='tanh')
    dropout = layers.Dropout(rate=drop_rate)
    dense_out = layers.Dense(units = n_y, activation='linear')
    concatenator = layers.Concatenate()
    adder = layers.Add()

    
    x_pdf_input = layers.Input(shape=(None, 1))
    x_x_input = layers.Input(shape=(None, 1))
    
    encoder_input = concatenator([x_pdf_input, x_x_input])

    encoder_output, for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c = encoder_LSTM(encoder_input)  
    dense_encoder_out = dense_encoder(encoder_output)

    decoder_input = layers.Input(shape=(1,n_a))
    decoder_state = [for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c]

    decoder_output = decoder_LSTM_L1(decoder_input, initial_state=decoder_state)
    decoder_output = decoder_LSTM_L2(decoder_output, initial_state=decoder_state)
    decoder_output_flat = flatter(decoder_output)
    out = dense_L1(decoder_output_flat)
    out = dropout(out)
    out = dense_L2(out)
    out = dropout(out)
    out = dense_out(out)
    out = adder([out, dense_encoder_out])
    
    model = keras.models.Model(inputs=[x_pdf_input, x_x_input, decoder_input], outputs=out)
    return model

def skipper_interp(n_a, n_dense, drop_rate, n_y):
    encoder_LSTM = layers.Bidirectional(layers.LSTM(units = n_a, return_state=True))
    decoder_LSTM_L1 = layers.Bidirectional(layers.LSTM(units = n_a, return_sequences=True))
    decoder_LSTM_L2 = layers.Bidirectional(layers.LSTM(units = n_a, return_sequences=True))
    flatter = layers.Flatten()
    dense_L1 = layers.Dense(units = n_dense, activation='tanh')
    dense_L2 = layers.Dense(units = n_dense, activation='tanh')
    dropout = layers.Dropout(rate=drop_rate)
    dense_out = layers.Dense(units = n_y, activation='linear')
    adder = layers.Add()
    input_reshapor = layers.Reshape((n_y,))

    
    x_pdf_input = layers.Input(shape=(n_y, 1))

    encoder_output, for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c = encoder_LSTM(x_pdf_input)  
    
    decoder_input = layers.Input(shape=(1,n_a))
    decoder_state = [for_encoder_h, for_encoder_c, back_encoder_h, back_encoder_c]

    decoder_output = decoder_LSTM_L1(decoder_input, initial_state=decoder_state)
    decoder_output = decoder_LSTM_L2(decoder_output, initial_state=decoder_state)
    decoder_output_flat = flatter(decoder_output)
    out = dense_L1(decoder_output_flat)
    out = dropout(out)
    out = dense_L2(out)
    out = dropout(out)
    out = dense_out(out)
    x = input_reshapor(x_pdf_input)
    out = adder([out, x])
    
    model = keras.models.Model(inputs=[x_pdf_input, decoder_input], outputs=out)
    return model

def feedforward(n_dense, drop_rate, n_y):
    dense_L1 = layers.Dense(units=n_dense)
    dense_L2 = layers.Dense(units=n_dense)
    dense_L3 = layers.Dense(units=n_dense)
    dense_out = layers.Dense(units=n_y)
    dropout = layers.Dropout(rate=drop_rate)


    input_x = layers.Input(shape=(n_y,))

    x = dense_L1(input_x)
    x = dense_L2(x)
    x = dropout(x)
    x = dense_L3(x)
    x = dropout(x)
    out = dense_out(x)

    model = Model(inputs=input_x, outputs=out)
    return model