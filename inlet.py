# coding=utf-8
from pylsl import StreamInlet, resolve_stream
from utils import epoch_buffer, sample_arr, filter
import time
from scipy import signal

def save_data():
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    # here the streams[0] means select the first stream in the streams
    inlet = StreamInlet(streams[0], max_buflen=60)
    start_state = True
    # global start_state
    while True:
        sample, timestamp = inlet.pull_chunk(timeout=0.0, max_samples=15000)

        if sample:
            if start_state:
                print("please waiting for 5s  the data to stabilize")
                time.sleep(5)
                start_state = False
            else:
                sample_list = sample_arr(sample)  # list
                filter_list = filter(sample_list)  # filter
                down_sample = signal.resample(filter_list, 100)  # down sample
                epoch_buffer.set_raw_data(filter_list)  # save filter data
                epoch_buffer.set_data(down_sample.tolist())  # save down sample
                # print(down_sample[0])
            time.sleep(1)

def main():
    save_data()
if __name__ == "__main__":
    main()