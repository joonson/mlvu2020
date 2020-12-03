# Lint as: python3
# Joon Son Chung
r"""Example using PyCoral to evaluate face embeddings.

The dataset format should be the same as Korean Faces in the Wild (validation and test sets),
with test_list.csv inside the folder.

Example usage:
```
python3 face_embedding.py \
  --model test_data/face.tflite  \
  --test_path test_data/val
```
"""

import argparse
import time
import numpy
import pdb
import os
import glob

from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from sklearn import metrics

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    
    return (tunedThreshold, eer, fpr, fnr);

def numpy_loader(filename):
  image = Image.open(filename)
  image = image.resize((256,256),resample=Image.BILINEAR)
  image = image.crop((16,16,240,240))
  image = numpy.asarray(image, dtype=numpy.float32) / 255.
  image = numpy.subtract(image, numpy.array([0.485,0.456,0.406]))
  image = numpy.divide(image, numpy.array([0.229,0.224,0.225]))
  image = numpy.expand_dims(image, 0) 
  return image

def set_input_tensor(interpreter, input):
  input_details = interpreter.get_input_details()[0]
  tensor_index = input_details['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  scale, zero_point = input_details['quantization']
  input_tensor[:, :] = numpy.uint8(input / scale + zero_point)

def main():

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-t', '--test_path', required=True,
                      help='Image to the dataset.')
  args = parser.parse_args()

  ## ========== ========== ===========
  ## Load the network
  ## ========== ========== ===========

  interpreter = make_interpreter(*args.model.split('@'))
  interpreter.allocate_tensors()


  ## ========== ========== ===========
  ## Compute embeddings
  ## ========== ========== ===========


  test_list = os.path.join(args.test_path,'test_list.csv')

  ## Read all lines
  with open(test_list) as f:
    lines = f.readlines()

  files = glob.glob(args.test_path+'/*/*.jpg')

  embeddings = {}

  for fidx, file in enumerate(files):

    start = time.perf_counter()

    image = numpy_loader(file)
    set_input_tensor(interpreter, image)
    
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details['index'])
    # Outputs from the TFLite model are uint8, so we dequantize the results:
    scale, zero_point = output_details['quantization']
    output = numpy.array(output, dtype=numpy.float32)
    output = scale * (output - zero_point)
    print('Computing feature %d - %.1fms' % (fidx, inference_time * 1000))

    embeddings[file] = output

  ## ========== ========== ===========
  ## Compute cosine distances
  ## ========== ========== ===========

  print('Evaluating ...')

  all_scores = []
  all_labels = []

  for idx, line in enumerate(lines):

    data = line.strip().split(',')

    ref_feat = embeddings[os.path.join(args.test_path,data[1])][0]
    com_feat = embeddings[os.path.join(args.test_path,data[2])][0]

    cos_sim = numpy.dot(com_feat, ref_feat)/(numpy.linalg.norm(com_feat)*numpy.linalg.norm(ref_feat))

    all_scores.append(cos_sim)
    all_labels.append(int(data[0]))

  ## ========== ========== ===========
  ## Compute EER
  ## ========== ========== ===========

  results = tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])

  print('EER is',results[1])


if __name__ == '__main__':
  main()
