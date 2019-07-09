import sys
import math
import random
import numpy as np
import random as pr
from rdkit import Chem
from make_smile import *
from keras.models import load_model
from keras.preprocessing import sequence
from load_model import loaded_model
def chem_kn_simulation(model,state,val):
    max_len = 82
    get_int = [val.index(state[j]) for j in range(len(state))]
    x       = np.reshape(get_int,(1,len(get_int)))
    x_pad   = sequence.pad_sequences(x, maxlen = max_len, dtype = 'int32', padding = 'post', truncating = 'pre', value = 0.0)

    while not get_int[-1] == val.index("\n"):
        predictions   = model.predict(x_pad)
        #print("shape of RNN",predictions.shape)
        a             = predictions[0][len(get_int) - 1]
        preds         = np.asarray(a).astype('float64')
        # preds         = np.log(preds) / 1.0
        # preds         = np.exp(preds)
        preds         = preds / np.sum(preds)
        next_probas   = np.random.multinomial(1, preds, 1)
        next_int      = np.argmax(next_probas)
        next_int_test = sorted(range(len(a)), key = lambda i: a[i])[-10:]
        get_int.append(next_int)
        x             = np.reshape(get_int,(1,len(get_int)))
        x_pad         = sequence.pad_sequences(x, maxlen = max_len, dtype = 'int32', padding='post', truncating='pre', value=0.0)
        if len(get_int) > max_len:
            break

    # print([get_int])
    return [get_int]


def predict_smile(all_posible,val):
    new_compound = []
    for i in range(len(all_posible)):
        generate_smile  = [val[all_posible[i][j]] for j in range(len(all_posible[i])-1)]
        # generate_smile.remove("&")
        new_compound.append(generate_smile)

    return new_compound


def make_input_smile(generate_smile):
    new_compound = []
    for i in range(len(generate_smile)):
        middle = [generate_smile[i][j] for j in range(len(generate_smile[i]))]
        com    = ''.join(middle)
        new_compound.append(com)

    return new_compound


### Example of using the above three functions to generate molecules
val = [ '\n', '&','C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F',
        '[C@@H]', 'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]',
        's', 'Br', '/', '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]',
        '[nH+]', '\\', '[S@]', '5', '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7',
        'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]',
        '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]',
        '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
valc = [ '\n', '&','C', '(', ')', 'c', '1', '2', '=','#']
if __name__ == "__main__":
    smiles         = zinc_data_with_bracket_original()
    val, all_smile = zinc_processed_with_bracket(smiles)
    print val
    all_smile[0].remove('&')
    all_smile[0].remove('\n')
    print(all_smile[0])
    #model          = load_model('/home/macenrola/Documents/MACHINE_LEARNING/ChemTS/ChemTS/RNN-model/model.h5')
    model =loaded_model()

    #all_posible    = chem_kn_simulation(model,all_smile[0],val)
    all_posible    = chem_kn_simulation(model,['C'],valc)
    generate_smile = predict_smile(all_posible,valc)
    new_compound = make_input_smile(generate_smile)

    print(new_compound)
