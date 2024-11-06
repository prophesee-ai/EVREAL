import math
import os
import json
import glob
import argparse
import numpy as np
import h5py




def h5_to_npy(h5_path, output_pth):
    metadata_path = os.path.join(output_pth, 'metadata.json')
    events_ts_path = os.path.join(output_pth, 'events_ts.npy')
    events_xy_path = os.path.join(output_pth, 'events_xy.npy')
    events_p_path = os.path.join(output_pth, 'events_p.npy')
    images_path = os.path.join(output_pth, 'images.npy')
    images_ts_path = os.path.join(output_pth, 'images_ts.npy')
    image_event_indices_path = os.path.join(output_pth, 'image_event_indices.npy')
    sensor_size = None
    image_list = []
    file = h5py.File(h5_path, 'r')
    for key in ['events', 'images', 'event_indices']:
        if key == 'events':
            xs = np.array(file['events/xs'])
            ys = np.array(file['events/ys'])
            ts = np.array(file['events/ts']) / 1e6  # convert to s
            ps = np.array(file['events/ps'])
        elif key == 'images':
            for image in file['images']:
                image = np.array(file['images'][image])
                if sensor_size is None:
                    sensor_size = image.shape[:2]
                elif sensor_size != image.shape[:2]:
                    raise ValueError("Warning: sensor size mismatch. Expected {}, got {}".format(sensor_size, image.shape[:2]))
                image_list.append(image)
            images = np.stack(image_list)
        elif key == 'event_indices':
            image_event_indices = np.array(file['event_indices'])
            # start of exposure
            images_ts = np.array([ts[idx[0]] for idx in image_event_indices])

    events_ts = ts
    events_xy = np.array([xs, ys]).transpose()
    events_p = ps

    # # If some timestamps are erroneous (decreasing), replace them with the average of the surrounding timestamps
    # # (Required for engineering_posters sequence from HQF dataset, where there is an error with images_ts[528])
    # mask = images_ts[:-1] > images_ts[1:]
    # avg_values = (images_ts[:-2] + images_ts[2:]) / 2.0
    # images_ts[1:-1][mask[:-1]] = np.squeeze(avg_values)[mask[:-1]]

    images = np.expand_dims(images, axis=-1)
    images_ts = np.expand_dims(images_ts, axis=1)

    img_min_ts = np.min(images_ts)
    events_min_ts = np.min(events_ts)
    min_ts = min(img_min_ts, events_min_ts)
    events_ts -= min_ts
    images_ts -= min_ts

    np.save(events_ts_path, events_ts, allow_pickle=False, fix_imports=False)
    np.save(events_xy_path, events_xy, allow_pickle=False, fix_imports=False)
    np.save(events_p_path, events_p, allow_pickle=False, fix_imports=False)

    np.save(images_path, images, allow_pickle=False, fix_imports=False)
    np.save(images_ts_path, images_ts, allow_pickle=False, fix_imports=False)
    np.save(image_event_indices_path, image_event_indices, allow_pickle=False, fix_imports=False)

    # write sensor size to metadata
    metadata = {"sensor_resolution": sensor_size}
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    """ Tool for converting h5 events and images to numpy format. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory of h5 files")
    parser.add_argument("--remove", help="Remove rosbags after conversion", action="store_true")
    args = parser.parse_args()
    h5_paths = sorted(glob.glob(os.path.join(args.path, "*.h5")))
    for path in h5_paths:
        print("Processing {}".format(path))
        output_pth = os.path.splitext(path)[0]
        os.makedirs(output_pth, exist_ok=True)
        h5_to_npy(path, output_pth)
        if args.remove:
            os.remove(path)
