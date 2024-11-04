import torch
import numpy as np

from cimaging_ml.processings.event_processing import compute_polarity_histograms
from metavision_sdk_base import EventCD


def events_to_image_torch(xs, ys, ps, device=None, sensor_size=(180, 240)):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
        :param xs: tensor of x coords of events
        :param ys: tensor of y coords of events
        :param ps: tensor of event polarities/weights
        :param device: the device on which the image is. If none, set to events device
        :param sensor_size: the size of the image sensor/output image
    """
    if device is None:
        device = xs.device

    img_size = list(sensor_size)

    img = torch.zeros(img_size).to(device)
    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)
    return img


def events_to_voxel_torch(xs, ys, ts, ps, num_bins, device=None, sensor_size=(180, 240)):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    num_bins : number of bins in output voxel grids (int)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert (len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps))
    bins = []
    dt = ts[-1] - ts[0]
    if dt.item() < 1e-9:
        t_norm = torch.linspace(0, num_bins - 1, steps=len(ts))
    else:
        t_norm = (ts - ts[0]) / dt * (num_bins - 1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(num_bins):
        bilinear_weights = torch.max(zeros, 1.0 - torch.abs(t_norm - bi))
        weights = ps * bilinear_weights
        vb = events_to_image_torch(xs, ys, weights, device, sensor_size=sensor_size)
        bins.append(vb)
    bins = torch.stack(bins)
    return bins

def events_to_polarity_fixed_bin_exposure_voxel_torch(
    histo_processor,
    xs,
    ys,
    ts,
    ps,
    device=None,
    sensor_size=(180, 240)
):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    Parameters
    ----------
    xs : list of event x coordinates (torch tensor)
    ys : list of event y coordinates (torch tensor)
    ts : list of event timestamps (torch tensor)
    ps : list of event polarities (torch tensor)
    device : device to put voxel grid. If left empty, same device as events
    sensor_size : the size of the event sensor/output voxels
    Returns
    -------
    voxel: voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert (len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps))
    dt = ts[-1] - ts[0]
    dt *= 1e9

    xs = xs.cpu().numpy().astype(dtype=np.int32)
    ys = ys.cpu().numpy().astype(dtype=np.int32)
    ps = ps.cpu().numpy().astype(dtype=np.int32)
    ts = (ts.cpu().numpy() * 1e9).astype(dtype=np.int32)

    # Create the structured array
    events = np.empty(xs.shape, dtype=EventCD)
    for i in range(len(events)):
        events[i]["x"] = xs[i]
        events[i]["y"] = ys[i]
        events[i]["p"] = ps[i] if ps[i] == 1 else 0
        events[i]["t"] = ts[i]

    hist = histo_processor.process(
        events,
        sensor_size[0],
        sensor_size[1],
        row_exposure_time=dt,
        exposure_map=None,
        start_exposure_first_line=ts[0],
        thoff=1.0,
        thon=1.0,
        sharp_ts_offset=None,
    )
    # hist = hist.clip(0, 100)

    histo_bin_size, histo_exposure_time, n_histograms = histo_processor.compute_histo_bin_size(
        dt,
        None
    )
    # latency model expect even number of time bins so add zero padding if needed
    if hist.shape[1] % 2 == 1:
        hist = np.concatenate([hist, np.zeros_like(hist[:, :1])], axis=1)
    return torch.tensor(hist, device=device, dtype=torch.float), histo_bin_size, histo_exposure_time
