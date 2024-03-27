from drivers.Serial import *
import cv2
import queue
from comps.utils import *
from comps.IRdataCollect import *
from comps.SonicDataCollect import *

myserial = None


class DEVICE:
    IR_data_collector = IRDataCollect()
    sonic_device1 = SonicDataCollect(MESSAGE.sonic1, 'sonic1')


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


'''
The key are mostly detected in IRdataCollect.playIR(), because the key is obtained by cv2
'''
def key_handler():
    while True:
        key = MESSAGE.KEY.get()
        if key == 32:
            # space
            CONTROL.RECORDING = not CONTROL.RECORDING
            print(f'recording now {CONTROL.RECORDING}')

            if not CONTROL.RECORDING and len(DEVICE.IR_data_collector.IR_imgs) > 0:
                args = CONTROL.get_scenetype()
                if args is False:
                    # discard data
                    DEVICE.IR_data_collector.clear_buffer()
                else:
                    DEVICE.IR_data_collector.save_data(*args)
                    # save some parameters to instance
                    CONTROL.update_lastround(args)
                    # another change for re-saving the files
                    print("上一轮的存储是否想改变主意？按下ESC以重新保存，否则请忽略。")

        elif key == 27:
            # esc, to re-save the last round file to another directory
            args = CONTROL.get_scenetype()
            if args is False:
                # discard data
                DEVICE.IR_data_collector.clear_buffer()
            else:
                new_filename = CONTROL.last_filename.replace(CONTROL.last_scenetype, args[0])
                args[1] = new_filename
                DEVICE.IR_data_collector.resave_data(*args)
                # save some parameters to instance
                CONTROL.update_lastround(args)
                # another change for re-saving the files
                print("上一轮的存储是否想改变主意？按下ESC以重新保存，否则请忽略。")

        elif key == ord('q'):
            break


def main():
    jobs = [message_classify, DEVICE.sonic_device1.play_sonic, DEVICE.IR_data_collector.play_IR]
    my_threads = []


    for job in jobs:
        t = threading.Thread(target=job,)
        my_threads.append(t)
        t.daemon = True
        t.start()

    print(f'System started with {len(jobs)} threads.')

    key_handler()
    # for t in my_threads:
    #     t.join()


if __name__ == '__main__':
    try:
        myserial = MySerial_2head1tail(b'\xFA', 'COM3', b'\xAF', b'\xFF')
        main()
    except Exception as e:
        myserial.portClose()
