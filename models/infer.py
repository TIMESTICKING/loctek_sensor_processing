import torch
import torch.nn as nn
import torch.nn.functional as F
from models.preprocess import *
from models.model import *
from comps.utils import *
import collections
from PyQt6.QtCore import QObject, pyqtSignal

class MyInference(QObject):
    predict_result_signal = pyqtSignal(list)
    def __init__(self) -> None:
        super().__init__()
        self.low_net = MyMLP().to(mydevice)
        self.high_net = MyMLP().to(mydevice)

        # load default model
        self.load_network_low_position('models/checkpoints_v2/low/F9-14-MLP221-Fea4-24-gap32-miniFilter-0d88.pth')
        self.load_network_high_position('models\checkpoints_v2\high\F9-14-MLP221-Fea4-24-gap32-miniFilter-0d86.pth')

        # default model is none, you need to specify one
        self.net: MyMLP = None 
        self.position = 0

        self.label = ['idle', 'sit', 'sit2stand', 'stand', 'stand2sit']
        self.action = ['下降', '不动', '升起']
        self.label_filter_size = 8
        self.predicted_label_raw = collections.deque(maxlen=self.label_filter_size)
        self.threadon = 0

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
        self.net = self.low_net if position == 0 else self.high_net

    @torch.no_grad()
    def get_label(self, IR_data, distance_data):
        IR_data = IR_data.to(mydevice)
        distance_data = distance_data.to(mydevice)

        outputs = self.net(IR_data, distance_data)
        outputs = outputs.cpu()

        _, predicted = torch.max(outputs.data, 1)

        return outputs.numpy(), self.label[predicted]


    def get_action(self):
        """Get the predicted label and the action of the table
        """        
        while self.net is None:
            print("请调用 set_table_position(position=<int 0 or 1>) 来设置当前桌子的高低")
            time.sleep(2)
        while True:
            if self.threadon == 1 and len(MESSAGE.IR_net_ready) == FRAME_IR and len(MESSAGE.sonic_net_ready) == FRAME_DISTANCE:
                IR_data, _, _ = scale_IR(np.array(MESSAGE.IR_net_ready))
                distance_data = torch.from_numpy(distance_preprocess(np.array(MESSAGE.sonic_net_ready, dtype=np.float32)))

                label_raw, label = self.get_label(IR_data, distance_data)
                # push into the queue
                self.predicted_label_raw.append(label_raw)
                # apply mean filter to the results in window size of 8
                mean_predicted_label_raw = np.mean(np.stack(self.predicted_label_raw), axis=0)
                filtered_label = int(np.argmax(mean_predicted_label_raw, 1))

                # get the action of the table from label and table position
                action = 0
                if self.position == 0:
                    action = 1 if filtered_label in [0, 1, 4] else 2
                else:
                    action = 1 if filtered_label in [0, 2, 3] else 0

                print(self.label[filtered_label], self.action[action])

                result = []
                result.append(self.label[filtered_label])
                result.append(self.action[action])
                self.predict_result_signal.emit(result)
                
                time.sleep(0.1)
            else:
                time.sleep(0.5)





if __name__ == '__main__':

    # prepare fake data
    pass




