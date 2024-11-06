"""
from https://github.com/Liu-haoyue/NER-Net/blob/main/Event_Trail_Suppression/ets.py
"""
import argparse
from tqdm import tqdm
import numpy as np
import torch


def ets_process(xs, ys, ts, ps, t0, s_w, s_h, threshold_t_on_us, threshold_t_off_us, soft_thr):
    ts *= 1e6  # Convert timestamps to us
    t0 *= 1e6  # Convert timestamps to us
    # ----------------------------Grid the events according to coordinates, with each pixel containing a sequence of timestamp values.----------------------------
    # Create two empty lists with a shape of [H, W].
    ts_map = [[[] for _ in range(s_w)] for _ in range(s_h)]
    p_map = [[[] for _ in range(s_w)] for _ in range(s_h)]

    # Traverse the array events and append t(i) to the list at the corresponding position in X.
    for i in range(len(xs)):
        ys_ = int(ys[i].item())
        xs_ = int(xs[i].item())
        ts_ = ts[i].item()
        ps_ = ps[i].item()
        ts_map[ys_][xs_].append(ts_)
        p_map[ys_][xs_].append(ps_)
    ts_map = np.array(ts_map)
    p_map = np.array(p_map)

    # Each element t_array in ts_map represents the timestamps of all events triggered at a pixel point (xx, yy). Convert the two-dimensional matrix into a one-dimensional array.
    ts_map = np.concatenate([np.array(row) for row in ts_map if len(row) > 0])
    p_map = np.concatenate([np.array(row) for row in p_map if len(row) > 0])

    # ----------------------------------------ETS processing----------------------------------------
    ets_events = np.ones((len(xs), 4)) * -1
    n_evs = 0

    for ii, t_array in enumerate(ts_map):
        # Skip elements that are empty lists.
        if not t_array:
            continue
        xx = ii % s_w
        yy = int((ii - xx) / s_w)
        t_array = np.array(t_array)
        if len(np.atleast_1d(t_array)) == 1:
            p_array = np.array(p_map[ii])
            ets_events[n_evs] = np.array([t_array, xx, yy, p_array])
            n_evs += 1
        else:
            sort_id = np.argsort(t_array)
            t_array = t_array[sort_id]
            p_array = np.array(p_map[ii])[sort_id]

            for nn in range(len(t_array)):
                if nn == 0:
                    num = 0
                    previous_p = p_array[nn]
                    previous_t = t_array[nn]
                    start_t = previous_t
                    time_interval = 0
                else:
                    if p_array[nn] == 1:
                        threshold_t = threshold_t_on_us
                    else:
                        threshold_t = threshold_t_off_us
                    # Events triggered within the same polarity, where the time interval since the last event is greater than the previous interval but less than the threshold value threshold_t.
                    if p_array[nn] == previous_p and t_array[nn] - previous_t > time_interval and t_array[nn] - previous_t < threshold_t:
                        # For events that meet the tailing condition, modify their triggering timestamps to be the time of the previous event triggered at that pixel plus 1 microsecond.
                        # Update iteration parameters.
                        num += 1
                        time_interval = t_array[nn] - previous_t - soft_thr
                        previous_t = t_array[nn]
                        t_array[nn] = start_t + num  # Correct timestamps.
                        # start_t = previous_t
                        previous_p = p_array[nn]
                    else:
                        # If the condition is not met, initialize parameters and start the next iteration
                        num = 0
                        previous_p = p_array[nn]
                        previous_t = t_array[nn]
                        start_t = previous_t
                        time_interval = 0

                ets_events[n_evs] = np.array([t_array[nn], xx, yy, p_array[nn]])
                n_evs += 1

    ets_events = ets_events.reshape(-1, 4)
    ets_events[:, 0] = ets_events[:, 0] + t0.item()
    # Reorder the events processed by ETS based on their timestamps
    idex = np.lexsort([ets_events[:, 0]])
    ets_events = ets_events[idex, :]
    # Release memory
    del ts_map, p_map
    xs = torch.tensor(ets_events[:, 1], dtype=torch.float)
    ys = torch.tensor(ets_events[:, 2], dtype=torch.float)
    ts = torch.tensor(ets_events[:, 0], dtype=torch.float) / 1e6
    ps = torch.tensor(ets_events[:, 3], dtype=torch.float)
    return xs, ys, ts, ps
