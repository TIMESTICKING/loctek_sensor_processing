from drivers.Serial import *
import cv2
import queue
from comps.utils import *
from comps.IRdataCollect import *

myserial = None

def message_classify():
    for res in myserial.readData():
        # print(res)
        paras = (res[1:], True, None)
        try:

            if res[0] == TAG.IR and not MESSAGE.IR.full():
                MESSAGE.IR.put(*paras)
            elif res[0] == TAG.SONIC1 and not MESSAGE.sonic1.full():
                MESSAGE.sonic1.put(*paras)
        except Exception as e:
            traceback.print_exc()
            break
            
            
def queue_watchdog():
    while True:
        print('IR queue length is: ', MESSAGE.IR.qsize())
        time.sleep(0.1)


def deal_input():
    while True:
        labels = int(input('Specify a scenetype from above ->'))
        print(labels)

        time.sleep(0.5)


def main():
    jobs = [message_classify]
    my_threads = []


    IR_data_collector = IRDataCollect()
    # jobs.append(IR_data_collector.play_IR)


    for job in jobs:
        t = threading.Thread(target=job,)
        my_threads.append(t)
        t.daemon = True
        t.start()

    print(f'System started with {len(jobs)} threads.')

    IR_data_collector.play_IR()
    # for t in my_threads:
    #     t.join()


if __name__ == '__main__':
    try:
        myserial = MySerial_2head1tail(b'\xFA', 'COM3', b'\xAF', b'\xFF')
        main()
    except Exception as e:
        myserial.portClose()
