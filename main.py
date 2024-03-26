from drivers.Serial import *
import cv2

from comps.utils import *
from comps.IRdataCollect import *

serial = MySerial_2head1tail(b'\xFA', 'COM3', b'\xAF', b'\xFF', 64 * 4)

def message_classify():
    for res in serial.readData():
        # print(res)
        paras = (res[1:], True, 0.5)
        try:

            if res[0] == TAG.IR:
                MESSAGE.IR.put(*paras)
            elif res[0] == TAG.SONIC1:
                MESSAGE.sonic1.put(*paras)

        except Exception as e:
            traceback.print_exc()
            
            

def main():
    jobs = [message_classify]
    my_threads = []


    IR_data_collector = IRDataCollect()
    jobs.append(IR_data_collector.play_IR)


    for job in jobs:
        t = threading.Thread(target=job,)
        my_threads.append(t)
        t.setDaemon(True)
        t.start()

    print(f'System started with {len(jobs)} threads.')

    for my_thread in my_threads:
        t.join()


if __name__ == '__main__':
    main()
