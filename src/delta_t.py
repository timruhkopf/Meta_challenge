"""
Looking at the readout behaviour of discretized, sparely & irregularly sampled
learning curves.
"""

def get_last_point_within_delta_t(C, delta_t):
    """"""
    temp_time = C + delta_t

    for i in range(len(timestamps)):
        if temp_time < timestamps[i]:
            if i == 0:  # if delta_t is not enough to get the first point, the agent wasted it for nothing!
                score, timestamp = 0.0, 0.0
            else:  # return the last achievable point
                score, timestamp = scores[i - 1], timestamps[i - 1]
            return(score, timestamp)

    # If the last point on the learning curve is already reached, return it
    score, timestamp = scores[-1], timestamps[-1]
    return(score, timestamp)

if __name__ == '__main__':
    import numpy as np

    # A discrete reperesentation of a learning curve
    timestamps = np.array([6, 54, 130, 194, 229, 328, 360])
    scores = np.array([0.23, 0.48, 0.75, 0.81, 0.81, 0.82, 0.82])

    # it does not suffice to jump to the next observed point!
    C = 54
    delta_t = 75
    print(get_last_point_within_delta_t(C, delta_t))

    # it does suffice to jump to the next point
    delta_t = 76
    print(get_last_point_within_delta_t(C, delta_t))

    # the over-next is reached
    delta_t = 140
    print(get_last_point_within_delta_t(C, delta_t))