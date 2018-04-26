import numpy as np
import os, sys
from cntk.ops.functions import load_model
from PIL import Image 
import time
import argparse




from IPython.display import Image as display 
from resizeimage import resizeimage


size = 32

label_lookup = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

z = load_model('./pred_vgg9.dnn')

arg_parser = argparse.ArgumentParser()	
arg_parser.add_argument("-d", "--image_dir", type=str, default="./", required=True,
	help="the directory to get test images")

args = vars(arg_parser.parse_args())
        
images_dir = f"{args['image_dir']}"

print(f"Image Directory: {images_dir}")

def evaluate_model(pred_op, image_path):
	label_lookup = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	st_time = time.clock()
	if image_path.find("png") >=0: return 0.0
	bgr_image = np.asarray(Image.open(image_path), dtype=np.float32) - 127.5
	bgr_image = bgr_image[..., [2, 1, 0]]
	pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
	
	result = np.squeeze(pred_op.eval({pred_op.arguments[0]:[pic]}))
	
	name =""
	names = ['Automobile','Airplane','Dog','Horse','Car','Ship','Truck','Bird']
	test_name = '_PDHCSTB'
	img_name = image_path[len(images_dir)+1:]
	name = names[test_name.find(img_name[0])]
    # Return top 3 results:
	top_count = 3
	result_indices = (-np.array(result)).argsort()[:top_count]
	end_time = time.clock()
	print("Top 3 predictions:")
	vowels ='oiuea'
	for i in range(top_count):
		label = label_lookup[result_indices[i]]
		conf = result[result_indices[i]] * 100
		article = 'an'
		if vowels.find(label_lookup[result_indices[i]][:1]) == -1:
			article ='a'
		if  result[result_indices[i]] * 100 >= 99.5:
			print("\tConfident {:s} {:8s}, confidence: {:.2f}%".format(article,label, conf))
		elif result[result_indices[i]] * 100 > 90.0:
			print("\tVery likely {:s} {:7s}, confidence: {:.2f}%".format(article,label, conf))
		else:
			print("\tLabel: {:14s}, confidence: {:.2f}%".format(label, conf))
		if name is "Car": name = "Automobile"
	print('Compute time was {:.10f}, and image was {:s}'.format(end_time-st_time,name))
	return (end_time-st_time)


def evaluate_image_dir(model, images_dir):
		ti = 0.0
		if not os.path.exists(images_dir):
			print(f"No such dir: {images_dir}")
			sys.exit(1)
		dirs = os.listdir(images_dir)
		print(f"Image Dir Content: {dirs}")
		for file in dirs:
			ti+=evaluate_model(model,f"{images_dir}/"+file)
		print('FPS:{:.4f}, {} frames can be processed per second.'.format((len(dirs)/ti),int(len(dirs)/ti)))
		
evaluate_image_dir(z,images_dir)
