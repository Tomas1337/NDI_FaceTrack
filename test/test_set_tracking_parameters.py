import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from TrackingModule import DetectionWidget
from random import seed
from random import random, randint

sys.path.insert(0, 'tool/')
from pipeclient import PipeClient 
from payloads import PipeClient_Parameter_Payload, Server_Payload, PipeClient_Image_Payload


def test_set_tracking_parameter():
    Tracker = DetectionWidget()
    parameter_list = Tracker.parameter_list
    custom_parameters = {}

    # Generate random variables
    seed(1)
    target_coordinate_x = randint(0,320*2)
    target_coordinate_y = randint(0,320*2)
    track_type = randint(0,3)   
    gamma = random()
    xMinE = random()
    yMinE = random()
    zoom_value = random()
    y_track_state = randint(0,1)
    autozoom_state = randint(0,1)
    reset_trigger = randint(0,1)

    # Generate dictionary
    for i in (parameter_list):
        custom_parameters[i] = locals()[i]
    
    Tracker.set_tracker_parameters(custom_parameters)
    tracker_parameters = Tracker.get_tracker_parameters()
    
    for key, value in custom_parameters.items():
        print(f'Parameter {key} is {tracker_parameters[key]} with value of {value}')
        assert (tracker_parameters[key] == value)

    print("Test done")



def test_changing_parameters():
    #TODO
    """
    Expected Behaviour:
    The custom parameters must have permanence in the Detection Widget
    If you set the custom parameter once, it must stay like this till the Tracker.set_tracker_parameter once again.

    """
if __name__ == '__main__':
    test_set_tracking_parameter()