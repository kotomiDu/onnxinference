import argparse
import numpy as np
import onnx
import time

np.set_printoptions(threshold=np.inf)


def test_unity_model(args):
    import onnxruntime
    import cv2
    im = cv2.imread(args.input_file)
    im = im[:,::-1,:]
    data = np.array(im,dtype=np.float32)
    data = data.reshape(1,3,448,448)
    session = onnxruntime.InferenceSession(args.model_file)
    prediction = session.run(None,{"input.1":data,"input.4":data, "input.7":data})
    #f = open("onnx_offset.txt",'w')

    d1,d2,d3,d4 = prediction[2].shape
    num1 = []
    for i in range(d1):
        for j in range(d2):
            for s in range(d3):
                for k in range(d4):
                    num1.append(prediction[2][i,j,s,k])
    num1 = np.array(num1)
    print("bigger than 0: {}".format(np.sum(num1 > 0)))
    print("small than 0: {}".format(np.sum(num1 < 0)))
    print("equal 1: {}".format(np.sum(num1 == 1)))
    print("equal -1: {}".format(np.sum(num1 == -1)))
    print("length:{}".format(len(num1)))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test onnx model')
    parser.add_argument('input_file')
    parser.add_argument('model_file')
    args = parser.parse_args()
 
    test_unity_model(args)
    
