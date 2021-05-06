from pyOpenBCI import OpenBCICyton
from pylsl import StreamInfo, StreamOutlet
import numpy as np

SCALE_FACTOR_EEG = (4500000.0) / 24 / (2 ** 23 - 1)  # uV/count 0.022351744455307063


def init(board):
    # the sample rate is 250hz, only turn on 1 channel
    # board.write_command('~6')
    # board.write_command('0')
    # board.write_command('2')
    board.write_command('3')
    board.write_command('4')
    board.write_command('5')
    board.write_command('6')
    board.write_command('7')
    board.write_command('8')
    board.write_command('!')
    # board.write_command('?')


print("Creating LSL stream for EEG. \nName: OpenBCIEEG\nID: OpenBCItestEEG\n")
info_eeg = StreamInfo('OpenBCIEEG', 'EEG', 8, 250, 'float32', 'OpenBCItestEEG')
outlet_eeg = StreamOutlet(info_eeg)
print("now sending data...")


def lsl_streamers(sample):
    print (np.array(sample.channels_data) * SCALE_FACTOR_EEG)

    outlet_eeg.push_sample(np.array(sample.channels_data) * SCALE_FACTOR_EEG)
    # print(np.array(sample.channels_data) * SCALE_FACTOR_EEG)



board = OpenBCICyton(port='/dev/ttyUSB0', daisy=False)



init(board)
print("start lsl steamers")
board.start_stream(lsl_streamers)
# print(sample.start_time, sample.board_type)
