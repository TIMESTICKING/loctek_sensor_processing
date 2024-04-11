import torch
import torch.nn as nn
import torch.nn.functional as F
from models.preprocess import *
from models.model import *
from comps.utils import *
import collections
from PyQt6.QtCore import QObject, pyqtSignal

class TableControl:
    def __init__(self):
        self.table_state = 0  # 0: Low, 1: High
        self.prev_pose = None
        self.stable_counter = 0
        self.stability_threshold = 3  # Number of consecutive frames required for stability

    def control_action(self, current_pose):
        action = 1  # Default action: 'no movement'

        if self.table_state == 0:  # Low state
            if current_pose in [0, 1, 4]:
                action = 1  # 'no movement'
            elif current_pose == 3:
                if self.prev_pose == 2:
                    action = 2  # 'rise'
                elif self.prev_pose == 3:
                    self.stable_counter += 1
                    if self.stable_counter >= self.stability_threshold:
                        action = 2  # 'rise' after stability
                else:
                    self.stable_counter = 0
            else:
                self.stable_counter = 0

        elif self.table_state == 1:  # High state
            if current_pose in [0, 2, 3]:
                action = 1  # 'no movement'
            elif current_pose == 1:
                if self.prev_pose == 4:
                    action = 0  # 'lower'
                elif self.prev_pose == 1:
                    self.stable_counter += 1
                    if self.stable_counter >= self.stability_threshold:
                        action = 1  # 'lower' after stability
                else:
                    self.stable_counter = 0
            else:
                self.stable_counter = 0

        # # Update the table state based on the action
        # if action == 1:
        #     self.table_state = 1
        # elif action == 2:
        #     self.table_state = 0

        self.prev_pose = current_pose
        return action


class MyInference(QObject):
    predict_result_signal = pyqtSignal(list)
    def __init__(self) -> None:
        super().__init__()
        self.low_net = MyMLP().to(mydevice)
        self.high_net = MyMLP().to(mydevice)

        # load default model
        self.load_network_low_position('models/checkpoints_v2/low/AllData_v2_balanced_0d86.pth')
        self.load_network_high_position('models\checkpoints_v2\high\AllData_v2_balanced_0d90.pth')

        # table control state machine
        self.table_controller = TableControl()

        # default model is none, you need to specify one
        self.net: MyMLP = None 
        self.position = 0

        self.label = ['idle', 'sit', 'sit2stand', 'stand', 'stand2sit']
        self.action = ['下降', '不动', '升起']

        self.predicted_label_q = collections.deque(maxlen=10)
        self.predicted_action_q = collections.deque(maxlen=5)

    def load_network_low_position(self, path):
        """给低位网络模型加载参数

        Args:
            path (string): 文件路径
        """        
        self.low_net.load_state_dict(torch.load(os.path.normpath(path), map_location=mydevice))

    def load_network_high_position(self, path):
        """给高位网络模型加载参数

        Args:
            path (string): 文件路径
        """        
        self.high_net.load_state_dict(torch.load(os.path.normpath(path), map_location=mydevice))


    def set_table_position(self, position=0):
        """设置桌子的位置

        Args:
            position (int): 0低位， 1高位. Defaults to 0.
        """        
        self.position = position
        self.table_controller.table_state = position
        self.net = self.low_net if position == 0 else self.high_net

    @torch.no_grad()
    def get_label(self, IR_data, distance_data):
        IR_data = IR_data.to(mydevice)
        distance_data = distance_data.to(mydevice)

        outputs = self.net(IR_data, distance_data)
        outputs = outputs.cpu()

        _, predicted = torch.max(outputs.data, 1)

        return outputs.numpy(), predicted
    

    def _filter_action(self, label, position):
        pass


    def get_action(self):
        """Get the predicted label and the action of the table
        """        
        while self.net is None:
            print("请调用 set_table_position(position=<int 0 or 1>) 来设置当前桌子的高低")
            time.sleep(2)
        while True:
            if len(MESSAGE.IR_net_ready) == FRAME_IR and len(MESSAGE.sonic_net_ready) == FRAME_DISTANCE:
                IR_data, _, _ = scale_IR(np.array(MESSAGE.IR_net_ready))
                distance_data = torch.from_numpy(distance_preprocess(np.array(MESSAGE.sonic_net_ready, dtype=np.float32)))

                label_raw, label = self.get_label(IR_data, distance_data)
                # push into the queue
                self.predicted_label_q.append(label[0])
                # apply mean filter to the results in window size of 8
                mode_filtered_label = np.argmax(np.bincount(self.predicted_label_q))
                # filtered_label = mode_predicted_label # int(np.argmax(mode_predicted_label, 1))

                # get the action of the table from label and table position
                action = self.table_controller.control_action(mode_filtered_label)

                # filter the action
                self.predicted_action_q.append(action)
                mode_predicted_action = np.argmax(np.bincount(self.predicted_action_q))

                # output the results
                print(self.label[mode_filtered_label], self.action[mode_predicted_action])
                self.predict_result_signal.emit([self.label[mode_filtered_label], self.action[mode_predicted_action]])
                
                time.sleep(0.1)
            else:
                time.sleep(0.5)





if __name__ == '__main__':

    # prepare fake data
    pass




