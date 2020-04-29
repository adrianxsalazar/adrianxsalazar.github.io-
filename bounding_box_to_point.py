from skimage import data
from skimage.viewer import ImageViewer
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import scipy.misc
import glob,os
import matplotlib.pyplot as plt
import cv2
import numpy as np


#class coco_image_extraction(Dataset). The class can be used to transform
#bounding box labels into point labels, save the point labels (all of them or
#jus the ones refering to a specific label), and extract the point labels
#from a given dataset
class box_to_point():

	def __init__(self,image_array,annotations_array,label_array,x1_y1_x2_y2=True):
		"""
		Args:
		    image_array: Array or list containing the images in array shape
		    annotations_array: Array or list containing the bounding boxes
			 					of each image
			label_array: Array or list containing the labels of each bounding boxes
						in each image
			x1_y1_x2_y2: True if the bounding boxes are in x1,y1,x2,y2 format
						False if the bounding boxes are in x1,y1,w,h format
		"""
		self.image_array=image_array
		self.annotations_array=annotations_array
		self.label_array=label_array
		self.point_array=np.array([self.box_to_point(annotations_array[i]) for i in range(len(annotations_array))])



	#function to transform a bounding box into a point in the center of the box
	#The box should be in format x1,y1,x2,y2
	def box_to_point(self,bounding_box_array, x1_y1_x2_y2=True):
		"""
		Args:
		    bounding_box_array: Array or list containing the bounding boxes
			 					of each image
			x1_y1_x2_y2: True if the bounding boxes are in x1,y1,x2,y2 format
						False if the bounding boxes are in x1,y1,w,h format
		"""


		#tool to transform the bounding box from  x, y, w, h to  x1,y1,x2,y2 format
		if x1_y1_x2_y2==False:
			x1,y1=bounding_box_array[:,0],bounding_box_array[:,1]
			x2=bounding_box_array[:.0]+bounding_box_array[:,2]
			y2=bounding_box_array[:,1]+bounding_box_array[:,3]
		else:
			#extract x1,y1,x2,y2
			x1,y1,x2,y2=bounding_box_array[:,0],bounding_box_array[:,1],bounding_box_array[:,2],bounding_box_array[:,3]

		#calculate the centers of the bounding box
		center_y=(y1+y2)/2
		center_y=center_y.astype(int)
		center_x=(x1+x2)/2
		center_x=center_x.astype(int)

		#create a unique array with the results
		center_array=np.vstack((center_x, center_y)).transpose()

		return center_array

	#Function to save the point labels and the images
	def create_pointdataset(self,images,boxes,labels,points_array,creation_path, label_filter=False):
		"""
		Args:
		    images: Array or list containing the images in array shape
		    boxes: Array or list containing the bounding boxes
			 					of each image
			labels: Array or list containing the labels of each bounding boxes
						in each image
			points_array: Array or list containing the ndicating points of
							each image
			creation_path: path to create the dataset
			label_filter: True to create multiple datasets, each with just the
							points of a speacif label

		"""
		#Create the directory where the data is going to be stored
		if not os.path.exists(creation_path+"/point_dataset"):
			os.makedirs(creation_path+"/point_dataset")


		#store the information in txt and png
		counter=0
		for i in range(len(images)):
			#Extract information
			image=images[i]
			label=labels[i]
			points=points_array[i]

			if label_filter == True:
				#create directory for the new discovere labels

				for l in np.unique(np.array(label)):

					if not os.path.exists(creation_path+"/point_dataset"+"/"+str(l)):
						os.makedirs(creation_path+"/point_dataset"+"/"+str(l))

					label_specific_list=[]

					for point_index in range(len(points)):
						point_label=label[point_index]
						single_point=points[point_index]

						if point_label == l:
							label_specific_list.append(single_point)

					txt_file_name=creation_path+"/point_dataset"+"/"+str(l)+"/"+str(counter)+".txt"
					file_name=creation_path+"/point_dataset"+"/"+str(l)+"/"+str(counter)+".png"

		 			scipy.misc.imsave(file_name,image)
					np.savetxt(txt_file_name,np.array(label_specific_list),fmt="%s")
			else:


				file_name=creation_path+"/point_dataset"+"/"+str(counter)+".png"
				txt_file_name=creation_path+"/point_dataset"+"/"+str(counter)+".txt"


	 			scipy.misc.imsave(file_name,image)
				np.savetxt(txt_file_name,points,fmt="%s")

			#counter +1
			counter=counter+1

	#function to load the images and txt files with the labels into an Array
	#containing the images and an array containing the labels
	def load_pointdataset(self,data_path, image_format=".png"):
		"""
		Args:
		    data_path: directory where the images and the txt are stored
		    image_format: extension or format of the images

		"""
		#create list to store the results
		image_array=[]
		point_array=[]

		#find the images and the txt file with the labels and put the information
		#into a label
		for file in os.listdir(data_path):
			if file.endswith(image_format):
				filename, file_extension = os.path.splitext(file)
				point_data= np.loadtxt(data_path+"/"+filename+".txt")
				image= scipy.misc.imread(data_path+"/"+filename+image_format)

				point_array.append(point_data)
				image_array.append(image)

		return image_array,point_array



	#function to display points
	def draw_points(self,image,points_array):
		"""
		Args:
		    image: image to be displayed
		    points_array=array containing the location of the points on the image
				we want to display
		"""

		implot = plt.imshow(image)
		plt.plot(points_array[:,0],points_array[:,1],'o')
		plt.show()


	#function to draw the boxes in an image
	def draw_boxes(self,image, boxes, labels):
		"""
		Args:
		    images: Array or list containing the image in array shape
		    boxes: Array or list containing the bounding boxes
			 					of the image to be displayed
			labels: Array or list containing the labels of each bounding boxes
						of the image that is going to be displayed
		"""

		# plot the image
		plt.imshow(image)
		# get the context for drawing boxes
		ax = pyplot.gca()
		# plot each box
		for i in range(len(boxes)):
			box = boxes[i]
			# get coordinates
			x1,y1, x2, y2 = box[0],box[1],box[2],box[3]
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='green')
			# draw the box
			ax.add_patch(rect)
			#draw text and score in top left corner
			label = "%s (%.3f)" % (labels[i])
			pyplot.text(x1, y1, label, color='white')
		# show the plot
		plt.show()

	#Function to display the image with the bounding boxes and the point labels
	def draw_points_and_boxes(self,image, boxes, labels, points_array):
		"""
		Args:
		    image: Array or list containing the images in array shape
		    boxes: Array or list containing the bounding boxes
			 					of the image
			labels: Array or list containing the labels of each bounding boxes
						in the image
			points_array: Array or list containing the ndicating points of
							the image

		"""

		# plot the image
		plt.imshow(image)
		# get the context for drawing boxes
		ax = pyplot.gca()
		# plot each box
		for i in range(len(boxes)):
			box = boxes[i]
			# get coordinates
			x1,y1, x2, y2 = box[0],box[1],box[2],box[3]
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='green')
			# draw the box
			ax.add_patch(rect)
			# draw text and score in top left corner
			label = str(labels[i])

			pyplot.text(x1, y1, label, color='white')
		plt.plot(points_array[:,0],points_array[:,1],'o')
		plt.show()

#Example
# from cocoparser import CocoDataset
# dataset = CocoDataset("./evaluation/testcocoformat/","RiseholmeSingle130")
# classes= dataset.classes
# label_dictionary=dataset.categories
# image_id=dataset.image_ids
# image_array=np.array([dataset.load_image(i) for i in range(len(image_id))])
# annotations_array=np.array([dataset.load_annotations(i)for i in range(len(image_id))])
# label_array=np.array([dataset.load_annotations(i)[:,4]for i in range(len(image_id))])
#
#
# dataset= box_to_point(image_array,annotations_array,label_array)
# dataset.draw_points_and_boxes(dataset.image_array[0],dataset.annotations_array[0],dataset.label_array[0],dataset.point_array[0])
