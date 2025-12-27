import MNN.numpy as np
import MNN
import MNN.cv as cv2

def inference(net, imgpath):
    """ inference ViT using a specific picture """
    # preprocess
    image = cv2.imread(imgpath)
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (224, 224)) / 255.
    #resize to mobile_net tensor size
    image = image - (0.48145466, 0.4578275, 0.40821073)
    image = image / (0.26862954, 0.26130258, 0.27577711)
    #change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)
    #Make var to save numpy; [h, w, c] -> [n, h, w, c]
    input_var = np.expand_dims(image, [0])
    #cv2 read shape is NHWC, Module's need is NC4HW4, convert it
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    #inference
    output_var = net.forward([input_var])

    predict = np.argmax(output_var[0])
    print("The image belongs to class: {}".format(predict))

# 模型加载
net = MNN.nn.load_module_from_file("./RL_detector_b.mnn", ["img"], ["prob"])
inference(net, "./imgs/fake.png")