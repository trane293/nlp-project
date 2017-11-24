from collections import namedtuple

# define bitstring as a list of zeros
bitstring = [0] * len(sentence)

# define a numedtuple which will be the state vector q
State = namedtuple('State', "e1 e2 bitstring r alpha")

# this is a container that will hold all states
state_holder = {}

# initial state
q0 = State('<s>', '<s>', bitstring, 0, 0)

# append initial state to state_holder
state_holder[0] = [q0]

# define bitstring as a list of zeros
bitstring = [0] * len(sentence)

# define a numedtuple which will be the state vector q
State = namedtuple('State', "e1 e2 bitstring r alpha")

# this is a container that will hold all states
state_holder = {}

# initial state
q0 = State('<s>', '<s>', bitstring, 0, 0)

# append initial state to state_holder
state_holder[0] = [q0]

def ph(q, d=4):
    ph_states = []
    for state in sc_P:
        flag = True  # we assume it as a valid state
        s = state[0]
        t = state[1]

        orig_bitstring = q.bitstring

        '''
        Step 1: Ensure bit string is not overlapped
        '''
        if s == t:
            # invalid state
            if orig_bitstring[s] != 0:
                flag = False
        else:
            # individial bits s and t are 0, but we also have to check in between them
            if orig_bitstring[s] == 0 and orig_bitstring[t] == 0:
                for _num in range(s, t + 1):
                    if orig_bitstring[_num] != 0:
                        flag = False
            else:
                flag = False

        '''
        Step 2: Ensure distortion limit is still obeyed
        '''

        r = q.r

        if not (abs(r + 1 - s) <= d):
            flag = False

        if flag == True:
            ph_states.append(state)

    return ph_states
