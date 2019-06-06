from enum import Enum

import numpy as np
from hmmlearn import hmm
from hmmlearn.base import _BaseHMM

from data_generator import generate_data

class Data(Enum):
    GENERATED_DATA = 0
    DEFAULT_DATA = 1

state_color = {
        'red': ['red'],
        'yellow': ['yellow'],
        'green': ['green'],
        'blinking_green': ['blue'],
        'yellow_red': ['yellow', 'red']
    }
states = ['red', 'yellow', 'green', 'blinking_green', 'yellow_red']
colors = ['red', 'yellow', 'green', 'black', 'yellow_red']

def get_model_generate(f):
    data = generate_data(f)

    probs = []
    for o in states:
        probs.append(sum(o == t[0] for t in data) / len(data))

    trans = [[0 for x in range(len(states))] for y in range(len(states))]
    states_i = np.zeros(len(states))
    for i in range(len(data) - 1):
        trans[states.index(data[i][0])][states.index(data[i + 1][0])] += 1
        states_i[states.index(data[i][0])] += 1

    p = data.pop()
    for i in range(len(states)):
        for j in range(len(states)):
            trans[i][j] /= states_i[i]
    data.append(p)

    probs_est = [[0 for x in range(len(states))] for y in range(len(colors))]
    for i in range(len(colors)):
        for j in range(len(states)):
            probs_est[i][j] = data.count((states[j], colors[i])) / sum(states[j] == t[0] for t in data)

    model = hmm.MultinomialHMM(len(states))
    model.n_features = len(colors)
    model.startprob_ = np.array(probs)
    model.transmat_ = np.array(trans)
    model.emissionprob_ = np.array(probs_est)
    #model = hmm.MultinomialHMM(len(states), np.array(probs), np.array(trans))
    #model = model.fit(np.array([colors.index(o[1]) for o in data]).reshape(-1, 1) )

    return model

def get_model_default():
    model = hmm.MultinomialHMM(len(states))
    model.n_features = len(colors)

    model.startprob_ = [0.3,  0.1, 0.5, 0.05, 0.05]

    model.transmat_ = np.array(
        [[0.75,  0.,         0. ,      0. ,         0.25 ],
         [0.1, 0.9, 0.,         0.,         0.,        ],
         [0.,         0.,         0.9,   0.1,   0.,        ],
         [0.,         0.01, 0.,         0.99, 0.,        ],
         [0.,         0.,         0.01, 0.,         0.99]]
    )

    model.emissionprob_ = np.array(
        [[0.85,        1.e-5,         1.e-5,        0.3,          0.05        ],
         [1.e-5,        0.9,          1.e-5,       0.2,          1.e-5        ],
         [1.e-5,          1.e-5,           0.7,       0.2,  1.e-5        ],
         [1.e-5,         1.e-5,          0.3,        0.4,  1.e-5        ],
         [0.2,         0.2,         1.e-5,         0.2,          0.9        ]]
    )


    # print(model.startprob_)
    # print(model.transmat_)
    # print(model.emissionprob_)

    return model

def get_model(fps, data_type):

    if data_type == Data.GENERATED_DATA:
        return get_model_generate(fps)
    else:
        return get_model_default()